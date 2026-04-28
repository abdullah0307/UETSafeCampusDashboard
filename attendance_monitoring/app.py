import json
import sqlite3
import html
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover - optional dependency in some envs
    st_autorefresh = None

from lab_survelliance.utils import (
    FaceRegistrationPage,
    SurveillanceRepository,
    SurveillanceService,
    _render_clickable_image_preview,
    _source_image_data_uri,
    _thumbnail_data_uri,
)
from utils.app_config import get_application_config
from utils.pagination import PaginationManager
from utils.theme_reset import clear_persisted_theme_once

class AttendanceHomePage:
    STATUS_STALE_AFTER_SECONDS = 10

    def __init__(
        self,
        attendance_root: Path,
        live_frames_dir: Path,
        stream_source_mode: str = "frame_files",
        streams_api_url: str | None = None,
        streams_public_base_url: str | None = None,
    ):
        self.attendance_root = attendance_root
        self.live_frames_root = live_frames_dir
        self.stream_source_mode = self._normalize_stream_source_mode(stream_source_mode)
        self.streams_api_url = str(streams_api_url or "").strip()
        self.streams_public_base_url = str(streams_public_base_url or "").strip()

    @staticmethod
    def _normalize_stream_source_mode(stream_source_mode: str | None) -> str:
        mode = str(stream_source_mode or "frame_files").strip().lower()
        aliases = {
            "live_frames_dir": "frame_files",
            "live_frames": "frame_files",
            "frame_files": "frame_files",
            "stream_api": "stream_api",
            "api": "stream_api",
        }
        return aliases.get(mode, "frame_files")

    def _is_stream_alive(self, payload: dict) -> bool:
        if not isinstance(payload, dict):
            return False

        status_value = str(payload.get("status") or "").strip().lower()
        if status_value != "running":
            return False

        last_update_text = str(payload.get("last_update") or "").strip()
        if not last_update_text:
            return False

        try:
            if "T" in last_update_text:
                last_update = datetime.fromisoformat(last_update_text.replace("Z", "+00:00"))
                current_time = datetime.now(last_update.tzinfo or timezone.utc)
            else:
                last_update = datetime.strptime(last_update_text, "%Y-%m-%d %H:%M:%S")
                current_time = datetime.now()
        except ValueError:
            return False

        age_seconds = (current_time - last_update).total_seconds()
        return age_seconds <= self.STATUS_STALE_AFTER_SECONDS

    def _streams_api_base_url(self) -> str:
        if not self.streams_api_url:
            return ""
        parts = urlsplit(self.streams_api_url)
        if not parts.scheme or not parts.netloc:
            return ""
        return f"{parts.scheme}://{parts.netloc}"

    def _load_stream_cards_from_files(self):
        if not self.live_frames_root.exists():
            return []

        cards = []
        for status_path in sorted(self.live_frames_root.glob("stream_*_status.json")):
            stream_slug = status_path.stem.replace("_status", "")
            frame_path = self.live_frames_root / stream_slug / "latest_frame.jpg"
            payload = None
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None

            alive = self._is_stream_alive(payload if isinstance(payload, dict) else {})
            payload_dict = payload if isinstance(payload, dict) else {}

            cards.append(
                {
                    "stream_id": payload_dict.get("stream_id"),
                    "stream_name": stream_slug.replace("_", " ").title(),
                    "status_path": status_path,
                    "frame_path": frame_path,
                    "video_url": "",
                    "payload": payload_dict,
                    "alive": alive,
                    "source_mode": "frame_files",
                }
            )

        return cards

    def _load_stream_cards_from_api(self):
        if not self.streams_api_url:
            return []

        try:
            response = requests.get(self.streams_api_url, timeout=5)
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []

        if isinstance(data, list):
            streams = data
        elif isinstance(data, dict):
            streams = data.get("streams", [])
        else:
            return []

        if not isinstance(streams, list):
            return []

        base_url = self._streams_api_base_url()
        cards = []
        for entry in streams:
            if not isinstance(entry, dict):
                continue

            payload = entry.get("status")
            if not isinstance(payload, dict):
                payload = {}

            stream_id = entry.get("stream_id")
            display_name = str(payload.get("display_name") or entry.get("display_name") or "").strip()
            if display_name:
                stream_name = display_name
            else:
                stream_name = f"Stream {stream_id}" if stream_id is not None else "Live Stream"
            cards.append(
                {
                    "stream_id": stream_id,
                    "stream_name": stream_name,
                    "status_path": None,
                    "frame_path": None,
                    "video_url": urljoin(base_url, str(entry.get("video_url") or "").strip()),
                    "status_url": urljoin(base_url, str(entry.get("status_url") or "").strip()),
                    "payload": payload,
                    "alive": self._is_stream_alive(payload),
                    "source_mode": "stream_api",
                }
            )

        return cards

    def _load_stream_cards(self):
        if self.stream_source_mode == "stream_api":
            return self._load_stream_cards_from_api()
        return self._load_stream_cards_from_files()

    def _streams_public_base(self) -> str:
        candidate = self.streams_public_base_url or self._streams_api_base_url()
        parts = urlsplit(candidate)
        if not parts.scheme or not parts.netloc:
            return ""
        return f"{parts.scheme}://{parts.netloc}"

    def _resolve_public_stream_url(self, video_url: str) -> str:
        video_url = str(video_url or "").strip()
        if not video_url:
            return ""

        public_base = self._streams_public_base()
        if not public_base:
            return video_url

        parsed = urlsplit(video_url)
        path = parsed.path or ""
        if parsed.query:
            path = f"{path}?{parsed.query}"

        if path:
            return urljoin(f"{public_base}/", path.lstrip("/"))
        return video_url

    def _stream_url_candidates(self, video_url: str) -> list[str]:
        direct_url = str(video_url or "").strip()
        public_url = self._resolve_public_stream_url(direct_url)
        candidates: list[str] = []
        for candidate in (direct_url, public_url):
            candidate = str(candidate or "").strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _frame_path_for_stream(self, stream_id) -> Path | None:
        try:
            stream_num = int(stream_id)
        except (TypeError, ValueError):
            return None
        return self.live_frames_root / f"stream_{stream_num}" / "latest_frame.jpg"

    def _render_stream_media(self, card: dict):
        if card.get("source_mode") == "stream_api":
            video_url = str(card.get("video_url") or "").strip()
            stream_urls = self._stream_url_candidates(video_url)
            if stream_urls:
                primary_url = html.escape(stream_urls[0], quote=True)
                safe_alt = html.escape(str(card.get("stream_name", "Live Stream")), quote=True)
                fallback_attr = ""
                if len(stream_urls) > 1:
                    fallback_url = html.escape(stream_urls[1], quote=True)
                    fallback_attr = (
                        "this.onerror=null;"
                        f"this.src='{fallback_url}';"
                    )
                st.markdown(
                    f"""
                    <div style="width:100%;background:#000;border-radius:0.5rem;overflow:hidden;line-height:0;">
                      <img
                        src="{primary_url}"
                        alt="{safe_alt}"
                        onerror="{fallback_attr}"
                        style="display:block;width:100%;height:auto;min-height:260px;object-fit:contain;background:#000;"
                      />
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return

            st.info("No live stream URL available.")
            return

        frame_path = card.get("frame_path")
        if isinstance(frame_path, Path) and frame_path.exists():
            st.image(str(frame_path), use_container_width=True)
        else:
            st.info("No live frame available.")

    def _render_stream_api_card(self, card: dict):
        payload = card.get("payload") or {}
        self._render_stream_media(card)

        metric_cols = st.columns(3)
        metric_cols[0].metric("Detected", int(payload.get("detected_count", 0) or 0))
        metric_cols[1].metric("Recognized", int(payload.get("recognized_count", 0) or 0))
        metric_cols[2].metric("FPS", f"{float(payload.get('fps', 0.0) or 0.0):.2f}")

        st.caption(
            f"Source: {payload.get('source', 'N/A')} | "
            f"Started: {payload.get('stream_started_at', 'N/A')} | "
            f"Last Update: {payload.get('last_update', 'N/A')}"
        )
        if card.get("status_url"):
            st.caption(f"Status URL: {card['status_url']}")

    def render(self):
        st.title("Attendance Management System")
        is_stream_api_mode = self.stream_source_mode == "stream_api"
        if is_stream_api_mode:
            st.caption("Live attendance stream overview using URL stream API.")
            controls = st.columns([1, 1, 1, 2])
            auto_refresh = controls[0].toggle(
                "Auto refresh",
                value=False,
                key="attendance_live_auto_refresh",
                help="Refresh live stream status periodically. Enabling this restarts MJPEG playback on each rerun.",
            )
            refresh_interval = controls[1].selectbox(
                "Interval",
                options=[5, 15, 30, 60],
                index=1,
                key="attendance_live_refresh_interval",
                disabled=not auto_refresh,
            )
            controls[2].button("Refresh now", key="attendance_live_refresh_now")
            if auto_refresh:
                if st_autorefresh is not None:
                    st_autorefresh(
                        interval=int(refresh_interval) * 1000,
                        key="attendance_live_status_refresh",
                    )
                else:
                    controls[3].caption("`streamlit-autorefresh` is unavailable in this environment.")
            else:
                controls[3].caption("Auto refresh is off.")

            stream_cards = [card for card in self._load_stream_cards_from_api() if card["alive"]]
            if not stream_cards:
                st.info("No live attendance streams were found.")
                return

            cols = st.columns(2)
            for idx, card in enumerate(stream_cards):
                with cols[idx % 2]:
                    with st.container(border=True):
                        st.subheader(card["stream_name"])
                        if card["alive"]:
                            st.success("stream is alive")
                        else:
                            st.error("stream is not alive")
                        self._render_stream_api_card(card)
            return

        st.caption("Live attendance stream overview using frame files.")
        controls = st.columns([1, 1, 1, 2])
        auto_refresh = controls[0].toggle(
            "Auto refresh",
            value=False,
            key="attendance_live_auto_refresh",
            help="Refresh live stream status periodically. Enabling this restarts MJPEG playback on each rerun.",
        )
        refresh_interval = controls[1].selectbox(
            "Interval",
            options=[5, 15, 30, 60],
            index=2,
            key="attendance_live_refresh_interval",
            disabled=not auto_refresh,
        )
        controls[2].button("Refresh now", key="attendance_live_refresh_now")
        if auto_refresh:
            if st_autorefresh is not None:
                st_autorefresh(
                    interval=int(refresh_interval) * 1000,
                    key="attendance_live_status_refresh",
                )
            else:
                controls[3].caption("`streamlit-autorefresh` is unavailable in this environment.")
        else:
            controls[3].caption("Auto refresh is off.")

        stream_cards = [card for card in self._load_stream_cards_from_files() if card["alive"]]
        if not stream_cards:
            st.info("No live attendance streams were found.")
            return

        cols = st.columns(2)
        for idx, card in enumerate(stream_cards):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.subheader(card["stream_name"])
                    if card["alive"]:
                        st.success("stream is alive")
                    else:
                        st.error("stream is not alive")

                    self._render_stream_media(card)
                    st.caption(
                        "Status JSON is missing, invalid, not running, or not updating."
                        if not card["alive"]
                        else (
                            f"Source: {card['payload'].get('source', 'N/A')} | "
                            f"Started: {card['payload'].get('stream_started_at', 'N/A')} | "
                            f"Last Update: {card['payload'].get('last_update', 'N/A')}"
                        )
                    )


class AttendanceLogsPage:
    def __init__(self, attendance_root: Path, activity_db_path: Path, table_name: str):
        self.attendance_root = attendance_root
        self.db_path = activity_db_path
        self.table_name = table_name

    def _load_logs(
        self,
        page: int = 1,
        per_page: int = 50,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()

        where_clauses: list[str] = []
        params: list = []

        if filters:
            username = filters.get("username")
            if username:
                where_clauses.append("username LIKE ?")
                params.append(f"%{username}%")

            start_date = filters.get("start_date")
            if start_date:
                where_clauses.append("date >= ?")
                params.append(str(start_date))

            end_date = filters.get("end_date")
            if end_date:
                where_clauses.append("date <= ?")
                params.append(str(end_date))

            stream_ids = filters.get("stream_id")
            if stream_ids:
                placeholders = ",".join("?" for _ in stream_ids)
                where_clauses.append(f"stream_id IN ({placeholders})")
                params.extend(stream_ids)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        offset = (page - 1) * per_page
        params.append(per_page)
        params.append(offset)

        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                f"""
                SELECT
                    id,
                    username,
                    date,
                    time,
                    stream_id,
                    image_path,
                    full_frame_image_path
                FROM {self.table_name}
                {where_sql}
                ORDER BY date DESC, time DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                conn,
                params=params,
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()

        return df

    def _load_logs_count(self, filters: dict | None = None) -> int:
        if not self.db_path.exists():
            return 0

        where_clauses: list[str] = []
        params: list = []

        if filters:
            username = filters.get("username")
            if username:
                where_clauses.append("username LIKE ?")
                params.append(f"%{username}%")

            start_date = filters.get("start_date")
            if start_date:
                where_clauses.append("date >= ?")
                params.append(str(start_date))

            end_date = filters.get("end_date")
            if end_date:
                where_clauses.append("date <= ?")
                params.append(str(end_date))

            stream_ids = filters.get("stream_id")
            if stream_ids:
                placeholders = ",".join("?" for _ in stream_ids)
                where_clauses.append(f"stream_id IN ({placeholders})")
                params.extend(stream_ids)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name} {where_sql}",
                params,
            )
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
        finally:
            conn.close()

    def render(self):
        st.title(f"{self.table_name.replace('_', ' ').title()}")
        st.caption(f"Search and review saved {self.table_name.replace('_', ' ')} records.")

        df = self._load_logs()
        if df.empty:
            st.info("No attendance_monitoring logs are available.")
            return

        df["username"] = df["username"].fillna("").astype(str).str.strip()
        df["date"] = df["date"].fillna("").astype(str).str.strip()
        df["time"] = df["time"].fillna("").astype(str).str.strip()
        df["stream_id"] = pd.to_numeric(df["stream_id"], errors="coerce").fillna(0).astype(int)
        df["logged_at"] = pd.to_datetime(
            df["date"] + " " + df["time"], errors="coerce"
        )
        df["image_path"] = df["image_path"].fillna("").astype(str)
        df["full_frame_image_path"] = df["full_frame_image_path"].fillna("").astype(str)

        valid_dates = df["logged_at"].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
        else:
            today = datetime.now().date()
            min_date = today
            max_date = today

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1.1, 1.1, 1, 1.4])
        with filter_col1:
            start_date = st.date_input(
                "From Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="attendance_logs_from_date",
            )
        with filter_col2:
            end_date = st.date_input(
                "To Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="attendance_logs_to_date",
            )
        with filter_col3:
            stream_options = sorted(df["stream_id"].dropna().unique().tolist())
            selected_streams = st.multiselect(
                "Streams",
                options=stream_options,
                default=stream_options,
                key="attendance_logs_stream_filter",
            )
        with filter_col4:
            search_text = st.text_input(
                "Search",
                value="",
                placeholder="Search by username",
                key="attendance_logs_search",
            ).strip()

        if start_date > end_date:
            st.warning("From Date cannot be after To Date.")
            return

        df = df[
            df["logged_at"].dt.date.between(start_date, end_date, inclusive="both")
        ].copy()

        if selected_streams:
            df = df[df["stream_id"].isin(selected_streams)].copy()

        if search_text:
            needle = search_text.lower()
            df = df[
                df["username"].str.lower().str.contains(needle, na=False)
                | df["image_path"].str.lower().str.contains(needle, na=False)
                | df["full_frame_image_path"].str.lower().str.contains(needle, na=False)
            ].copy()

        if df.empty:
            st.info("No attendance_monitoring logs match the selected filters.")
            return

        df = df.sort_values("logged_at", ascending=False).reset_index(drop=True)

        total_count = len(df)
        paginator = PaginationManager('attendance_logs', total_count, default_per_page=50)

        start_idx = paginator.offset
        end_idx = paginator.offset + paginator.per_page
        page_df = df.iloc[start_idx:end_idx]

        page_df["face_thumb"] = page_df["image_path"].apply(lambda p: _thumbnail_data_uri(p, max_side=220))
        page_df["frame_thumb"] = page_df["full_frame_image_path"].apply(
            lambda p: _thumbnail_data_uri(p, max_side=220)
        )

        st.caption(f"Showing {page_df.shape[0]} of {total_count} {self.table_name.replace('_', ' ')} record(s).")

        header_cols = st.columns([2.1, 2.3, 1.2, 2.7, 2.7])
        with header_cols[0]:
            st.markdown("**Username**")
        with header_cols[1]:
            st.markdown("**Logged At**")
        with header_cols[2]:
            st.markdown("**Stream**")
        with header_cols[3]:
            st.markdown("**Face Crop**")
        with header_cols[4]:
            st.markdown("**Screenshot**")

        records_container = st.container(height=760, border=False)
        with records_container:
            for row in page_df.itertuples(index=False):
                row_cols = st.columns([2.1, 2.3, 1.2, 2.7, 2.7], vertical_alignment="center")
                with row_cols[0]:
                    st.markdown(str(row.username or "Unknown"))
                with row_cols[1]:
                    st.markdown(
                        row.logged_at.strftime("%Y-%m-%d %H:%M:%S")
                        if pd.notna(row.logged_at)
                        else f"{row.date} {row.time}".strip()
                    )
                with row_cols[2]:
                    st.markdown(f"Stream {int(row.stream_id)}")
                with row_cols[3]:
                    _render_clickable_image_preview(
                        row.face_thumb,
                        _source_image_data_uri(row.image_path),
                        f"{row.username} • Face Crop • Stream {int(row.stream_id)}",
                        frame_height=190,
                        image_width=220,
                        image_height=160,
                    )
                with row_cols[4]:
                    _render_clickable_image_preview(
                        row.frame_thumb,
                        _source_image_data_uri(row.full_frame_image_path),
                        f"{row.username} • Screenshot • Stream {int(row.stream_id)}",
                        frame_height=190,
                        image_width=220,
                        image_height=160,
                    )
                st.divider()

        if paginator.render_pagination_controls():
            st.rerun()


class AttendanceDashboardPage:
    def __init__(self, attendance_root: Path, activity_db_path: Path, table_name: str):
        self.attendance_root = attendance_root
        self.db_path = activity_db_path
        self.table_name = table_name

    def _load_logs(
        self,
        page: int = 1,
        per_page: int = 50,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()

        where_clauses: list[str] = []
        params: list = []

        if filters:
            username = filters.get("username")
            if username:
                where_clauses.append("username LIKE ?")
                params.append(f"%{username}%")

            start_date = filters.get("start_date")
            if start_date:
                where_clauses.append("date >= ?")
                params.append(str(start_date))

            end_date = filters.get("end_date")
            if end_date:
                where_clauses.append("date <= ?")
                params.append(str(end_date))

            stream_ids = filters.get("stream_id")
            if stream_ids:
                placeholders = ",".join("?" for _ in stream_ids)
                where_clauses.append(f"stream_id IN ({placeholders})")
                params.extend(stream_ids)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        offset = (page - 1) * per_page
        params.append(per_page)
        params.append(offset)

        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                f"""
                SELECT
                    id,
                    username,
                    date,
                    time,
                    stream_id,
                    image_path,
                    full_frame_image_path
                FROM {self.table_name}
                {where_sql}
                ORDER BY date DESC, time DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                conn,
                params=params,
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()

        return df

    def _load_logs_count(self, filters: dict | None = None) -> int:
        if not self.db_path.exists():
            return 0

        where_clauses: list[str] = []
        params: list = []

        if filters:
            username = filters.get("username")
            if username:
                where_clauses.append("username LIKE ?")
                params.append(f"%{username}%")

            start_date = filters.get("start_date")
            if start_date:
                where_clauses.append("date >= ?")
                params.append(str(start_date))

            end_date = filters.get("end_date")
            if end_date:
                where_clauses.append("date <= ?")
                params.append(str(end_date))

            stream_ids = filters.get("stream_id")
            if stream_ids:
                placeholders = ",".join("?" for _ in stream_ids)
                where_clauses.append(f"stream_id IN ({placeholders})")
                params.extend(stream_ids)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name} {where_sql}",
                params,
            )
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
        finally:
            conn.close()

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df["username"] = df["username"].fillna("").astype(str).str.strip()
        df["date"] = df["date"].fillna("").astype(str).str.strip()
        df["time"] = df["time"].fillna("").astype(str).str.strip()
        df["stream_id"] = pd.to_numeric(df["stream_id"], errors="coerce").fillna(0).astype(int)
        df["logged_at"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
        df["hour"] = df["logged_at"].dt.hour.fillna(0).astype(int)
        df["date_only"] = df["logged_at"].dt.date

        return df

    def _apply_filters(
        self, df: pd.DataFrame, start_date, end_date, selected_streams, search_text
    ) -> pd.DataFrame:
        if df.empty:
            return df

        df = df[
            df["logged_at"].dt.date.between(start_date, end_date, inclusive="both")
        ].copy()

        if selected_streams:
            df = df[df["stream_id"].isin(selected_streams)].copy()

        if search_text:
            needle = search_text.lower()
            df = df[
                df["username"].str.lower().str.contains(needle, na=False)
            ].copy()

        return df

    def _render_metrics(self, df: pd.DataFrame):
        if df.empty:
            st.info("No data available for the selected filters.")
            return

        total_records = len(df)
        unique_users = df["username"].nunique()
        today = datetime.now().date()
        today_records = len(df[df["date_only"] == today])
        streams_count = df["stream_id"].nunique()

        metric_cols = st.columns(4)
        metric_cols[0].metric("Total Records", f"{total_records:,}")
        metric_cols[1].metric("Unique Users", f"{unique_users:,}")
        metric_cols[2].metric("Today's Attendance", f"{today_records:,}")
        metric_cols[3].metric("Active Streams", f"{streams_count:,}")

    def _render_timeline_chart(self, df: pd.DataFrame):
        if df.empty:
            return

        st.subheader("Attendance Timeline")

        daily_counts = df.groupby("date_only").size().reset_index(name="count")
        daily_counts = daily_counts.sort_values("date_only")

        fig = px.line(
            daily_counts,
            x="date_only",
            y="count",
            markers=True,
            labels={"date_only": "Date", "count": "Attendance Count"},
            title="Daily Attendance Trend",
        )
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    def _render_stream_distribution(self, df: pd.DataFrame):
        if df.empty:
            return

        st.subheader("Attendance by Stream")

        stream_counts = df.groupby("stream_id").size().reset_index(name="count")
        stream_counts = stream_counts.sort_values("count", ascending=False)

        fig = px.bar(
            stream_counts,
            x="stream_id",
            y="count",
            labels={"stream_id": "Stream ID", "count": "Attendance Count"},
            title="Attendance Distribution Across Streams",
            color="count",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=400, xaxis_title="Stream ID", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    def _render_hourly_activity(self, df: pd.DataFrame):
        if df.empty:
            return

        st.subheader("Hourly Activity Pattern")

        hourly_counts = df.groupby("hour").size().reset_index(name="count")
        hourly_counts = hourly_counts.sort_values("hour")

        fig = px.bar(
            hourly_counts,
            x="hour",
            y="count",
            labels={"hour": "Hour of Day", "count": "Attendance Count"},
            title="Attendance by Hour of Day",
            color="count",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            height=400,
            xaxis_title="Hour",
            yaxis_title="Count",
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_top_attendees(self, df: pd.DataFrame):
        if df.empty:
            return

        st.subheader("Top Attendees")

        user_counts = df.groupby("username").size().reset_index(name="count")
        user_counts = user_counts.sort_values("count", ascending=False).head(10)

        if user_counts.empty:
            st.info("No attendee data available.")
            return

        fig = px.bar(
            user_counts,
            x="count",
            y="username",
            orientation="h",
            labels={"username": "Username", "count": "Attendance Count"},
            title="Top 10 Users by Attendance",
            color="count",
            color_continuous_scale="Greens",
        )
        fig.update_layout(height=500, xaxis_title="Count", yaxis_title="Username")
        st.plotly_chart(fig, use_container_width=True)

    def render(self):
        st.title("Attendance Dashboard")
        st.caption("Analytics and insights for attendance_monitoring data")

        df = self._load_logs()
        if df.empty:
            st.info("No attendance_monitoring logs are available.")
            return

        df = self._prepare_data(df)

        valid_dates = df["logged_at"].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
        else:
            today = datetime.now().date()
            min_date = today
            max_date = today

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(
            [1.1, 1.1, 1, 1.4]
        )
        with filter_col1:
            start_date = st.date_input(
                "From Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="dashboard_from_date",
            )
        with filter_col2:
            end_date = st.date_input(
                "To Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="dashboard_to_date",
            )
        with filter_col3:
            stream_options = sorted(df["stream_id"].dropna().unique().tolist())
            selected_streams = st.multiselect(
                "Streams",
                options=stream_options,
                default=stream_options,
                key="dashboard_stream_filter",
            )
        with filter_col4:
            search_text = st.text_input(
                "Search",
                value="",
                placeholder="Search by username",
                key="dashboard_search",
            ).strip()

        if start_date > end_date:
            st.warning("From Date cannot be after To Date.")
            return

        filtered_df = self._apply_filters(
            df, start_date, end_date, selected_streams, search_text
        )

        self._render_metrics(filtered_df)

        if not filtered_df.empty:
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                self._render_timeline_chart(filtered_df)
            with chart_col2:
                self._render_stream_distribution(filtered_df)

            chart_col3, chart_col4 = st.columns(2)
            with chart_col3:
                self._render_hourly_activity(filtered_df)
            with chart_col4:
                self._render_top_attendees(filtered_df)


class PersonAnalyticsPage:
    INVALID_NAMES = {"", "unknown", "n/a", "none", "null", "nan"}

    def __init__(self, attendance_root: Path, activity_db_path: Path, table_name: str):
        self.attendance_root = attendance_root
        self.db_path = activity_db_path
        self.table_name = table_name

    def _load_logs(self) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                f"""
                SELECT
                    id,
                    username,
                    date,
                    time,
                    stream_id,
                    image_path,
                    full_frame_image_path
                FROM {self.table_name}
                ORDER BY date DESC, time DESC, id DESC
                """,
                conn,
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()

        if df.empty:
            return df

        df = df.copy()
        df["username"] = df["username"].fillna("").astype(str).str.strip()
        df = df[~df["username"].str.lower().isin(self.INVALID_NAMES)].copy()
        df["date"] = df["date"].fillna("").astype(str).str.strip()
        df["time"] = df["time"].fillna("").astype(str).str.strip()
        df["stream_id"] = pd.to_numeric(df["stream_id"], errors="coerce").fillna(0).astype(int)
        df["image_path"] = df["image_path"].fillna("").astype(str).str.strip()
        df["full_frame_image_path"] = (
            df["full_frame_image_path"].fillna("").astype(str).str.strip()
        )
        df["timestamp"] = pd.to_datetime(
            df["date"] + " " + df["time"], errors="coerce"
        )
        df = df[df["timestamp"].notna()].copy()
        if df.empty:
            return df

        df["date_only"] = df["timestamp"].dt.date
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        return df

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(int(round(float(seconds or 0))), 0)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes > 0:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    @staticmethod
    def _display_name_map(usernames: pd.Series) -> dict[str, str]:
        display_name_by_key: dict[str, str] = {}
        for raw_name in usernames.dropna().astype(str).str.strip():
            if not raw_name:
                continue
            key = raw_name.lower()
            current = display_name_by_key.get(key)
            if current is None or (raw_name[:1].isupper() and not current[:1].isupper()):
                display_name_by_key[key] = raw_name
        return display_name_by_key

    def _build_sessions(self, df: pd.DataFrame, gap_threshold_sec: int) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        gap_threshold_sec = max(int(gap_threshold_sec), 1)
        session_rows: list[dict] = []

        for person_key, group in df.groupby(df["username"].str.lower(), sort=False):
            person_df = group.sort_values("timestamp").reset_index(drop=True)
            session_start_idx = 0

            for idx in range(1, len(person_df) + 1):
                is_last = idx == len(person_df)
                if not is_last:
                    prev_ts = pd.Timestamp(person_df.loc[idx - 1, "timestamp"])
                    current_ts = pd.Timestamp(person_df.loc[idx, "timestamp"])
                    gap_seconds = (current_ts - prev_ts).total_seconds()
                if is_last or gap_seconds > gap_threshold_sec:
                    session_df = person_df.iloc[session_start_idx:idx].copy()
                    start_ts = pd.Timestamp(session_df["timestamp"].min())
                    end_ts = pd.Timestamp(session_df["timestamp"].max())
                    duration_sec = max((end_ts - start_ts).total_seconds(), 0.0)
                    preview_row = session_df.iloc[0]
                    preview_path = str(preview_row.get("image_path", "")).strip()
                    if not preview_path:
                        preview_path = str(
                            preview_row.get("full_frame_image_path", "")
                        ).strip()
                    stream_ids = sorted(
                        {int(stream_id) for stream_id in session_df["stream_id"].tolist()}
                    )
                    session_rows.append(
                        {
                            "person_key": person_key,
                            "username": str(session_df.iloc[0]["username"]),
                            "date_only": start_ts.date(),
                            "session_number": len(
                                [
                                    row
                                    for row in session_rows
                                    if row["person_key"] == person_key
                                ]
                            )
                            + 1,
                            "start_time": start_ts,
                            "end_time": end_ts,
                            "duration_sec": duration_sec,
                            "duration_label": self._format_duration(duration_sec),
                            "detection_count": len(session_df),
                            "stream_ids": stream_ids,
                            "stream_label": ", ".join(
                                f"Stream {stream_id}" for stream_id in stream_ids
                            ),
                            "preview_path": preview_path,
                            "preview_thumb": _thumbnail_data_uri(
                                preview_path, max_side=220
                            ),
                        }
                    )
                    session_start_idx = idx

        sessions_df = pd.DataFrame(session_rows)
        if sessions_df.empty:
            return sessions_df
        sessions_df = sessions_df.sort_values(
            ["username", "start_time"], ascending=[True, True]
        ).reset_index(drop=True)
        return sessions_df

    def _render_session_insights(self, person_sessions_df: pd.DataFrame):
        st.markdown("#### Session Insights")
        if person_sessions_df.empty:
            st.info("No sessions are available for the selected filters.")
            return

        sessions_df = person_sessions_df.copy()
        sessions_df["session_label"] = (
            "Session " + sessions_df["session_number"].astype(str)
        )
        sessions_df["start_label"] = sessions_df["start_time"].dt.strftime("%H:%M:%S")
        sessions_df["end_label"] = sessions_df["end_time"].dt.strftime("%H:%M:%S")

        insights_col1, insights_col2 = st.columns(2)
        with insights_col1:
            duration_fig = px.bar(
                sessions_df,
                x="session_label",
                y="duration_sec",
                color="stream_label",
                text="duration_label",
                custom_data=["start_label", "end_label", "detection_count"],
                labels={
                    "session_label": "Session",
                    "duration_sec": "Duration (sec)",
                    "stream_label": "Stream",
                },
                title="Session Durations",
            )
            duration_fig.update_traces(
                hovertemplate=(
                    "%{x}<br>"
                    "Duration: %{text}<br>"
                    "Start: %{customdata[0]}<br>"
                    "End: %{customdata[1]}<br>"
                    "Detections: %{customdata[2]}<extra></extra>"
                )
            )
            duration_fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(duration_fig, use_container_width=True)

        with insights_col2:
            stream_mix_df = (
                sessions_df.groupby("stream_label", as_index=False)
                .agg(
                    total_duration_sec=("duration_sec", "sum"),
                    session_count=("session_number", "count"),
                )
                .sort_values("total_duration_sec", ascending=False)
            )
            stream_mix_df["duration_label"] = stream_mix_df["total_duration_sec"].apply(
                self._format_duration
            )
            stream_mix_fig = px.pie(
                stream_mix_df,
                names="stream_label",
                values="total_duration_sec",
                hole=0.45,
                custom_data=["session_count", "duration_label"],
                title="Session Share by Stream",
            )
            stream_mix_fig.update_traces(
                hovertemplate=(
                    "Stream: %{label}<br>"
                    "Estimated Time: %{customdata[1]}<br>"
                    "Sessions: %{customdata[0]}<extra></extra>"
                )
            )
            stream_mix_fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(stream_mix_fig, use_container_width=True)

    def _render_person_metrics(
        self, person_df: pd.DataFrame, person_sessions_df: pd.DataFrame
    ):
        first_detection = person_df["timestamp"].min()
        last_detection = person_df["timestamp"].max()
        streams_seen = person_df["stream_id"].nunique()
        total_sessions = len(person_sessions_df)
        total_time_sec = (
            float(person_sessions_df["duration_sec"].sum())
            if not person_sessions_df.empty
            else 0.0
        )

        metric_cols = st.columns(6)
        metric_cols[0].metric(
            "First Detection",
            first_detection.strftime("%H:%M:%S") if pd.notna(first_detection) else "N/A",
        )
        metric_cols[1].metric(
            "Last Detection",
            last_detection.strftime("%H:%M:%S") if pd.notna(last_detection) else "N/A",
        )
        metric_cols[2].metric("Detection Count", f"{len(person_df):,}")
        metric_cols[3].metric("Total Sessions", f"{total_sessions:,}")
        metric_cols[4].metric("Estimated Total Time", self._format_duration(total_time_sec))
        metric_cols[5].metric("Streams Seen", f"{streams_seen:,}")

    def _render_timeline(self, person_sessions_df: pd.DataFrame, selected_name: str):
        st.markdown("#### Presence Timeline")
        if person_sessions_df.empty:
            st.info("No merged sessions are available for the selected filters.")
            return

        timeline_df = person_sessions_df.copy()
        timeline_df["person_row"] = selected_name
        timeline_df["hover_label"] = (
            "Session "
            + timeline_df["session_number"].astype(str)
            + " | "
            + timeline_df["duration_label"].astype(str)
        )

        fig = px.timeline(
            timeline_df,
            x_start="start_time",
            x_end="end_time",
            y="person_row",
            color="stream_label",
            custom_data=["hover_label", "detection_count", "stream_label"],
        )
        fig.update_yaxes(autorange="reversed", title_text="")
        fig.update_xaxes(title_text="Time")
        fig.update_traces(
            hovertemplate=(
                "%{customdata[0]}<br>"
                "Start: %{base|%Y-%m-%d %H:%M:%S}<br>"
                "End: %{x|%Y-%m-%d %H:%M:%S}<br>"
                "Detections: %{customdata[1]}<br>"
                "Streams: %{customdata[2]}<extra></extra>"
            )
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    def _render_daily_records(
        self,
        person_all_df: pd.DataFrame,
        person_all_sessions_df: pd.DataFrame,
        selected_name: str,
    ):
        st.markdown("#### Daily Records")
        if person_all_df.empty:
            st.info("No daily records are available for the selected person.")
            return

        detections_summary = (
            person_all_df.groupby("date_only", as_index=False)
            .agg(
                first_seen=("timestamp", "min"),
                last_seen=("timestamp", "max"),
                detections=("id", "count"),
                streams_seen=("stream_id", "nunique"),
                face_image_path=(
                    "image_path",
                    lambda values: next(
                        (str(value).strip() for value in values if str(value).strip()),
                        "",
                    ),
                ),
                screenshot_path=(
                    "full_frame_image_path",
                    lambda values: next(
                        (str(value).strip() for value in values if str(value).strip()),
                        "",
                    ),
                ),
            )
        )

        if not person_all_sessions_df.empty:
            sessions_summary = person_all_sessions_df.groupby("date_only", as_index=False).agg(
                sessions=("session_number", "count"),
                estimated_total_time_sec=("duration_sec", "sum"),
                stream_labels=("stream_label", lambda values: ", ".join(sorted(set(values)))),
            )
            daily_df = detections_summary.merge(
                sessions_summary, on="date_only", how="left"
            )
        else:
            daily_df = detections_summary.copy()
            daily_df["sessions"] = 0
            daily_df["estimated_total_time_sec"] = 0.0
            daily_df["stream_labels"] = ""

        daily_df["sessions"] = daily_df["sessions"].fillna(0).astype(int)
        daily_df["estimated_total_time_sec"] = daily_df[
            "estimated_total_time_sec"
        ].fillna(0.0)
        daily_df["estimated_total_time"] = daily_df["estimated_total_time_sec"].apply(
            self._format_duration
        )
        daily_df["first_seen_label"] = daily_df["first_seen"].dt.strftime("%H:%M:%S")
        daily_df["last_seen_label"] = daily_df["last_seen"].dt.strftime("%H:%M:%S")
        daily_df["date_label"] = pd.to_datetime(daily_df["date_only"]).dt.strftime(
            "%Y-%m-%d"
        )
        daily_df["streams_display"] = daily_df["stream_labels"].where(
            daily_df["stream_labels"].astype(bool),
            daily_df["streams_seen"].apply(
                lambda count: f"{int(count)} stream" if int(count) == 1 else f"{int(count)} streams"
            ),
        )
        daily_df["face_thumb"] = daily_df["face_image_path"].apply(
            lambda p: _thumbnail_data_uri(p, max_side=220)
        )
        daily_df["frame_thumb"] = daily_df["screenshot_path"].apply(
            lambda p: _thumbnail_data_uri(p, max_side=220)
        )
        daily_df = daily_df.sort_values("date_only", ascending=False).reset_index(drop=True)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Days Present", f"{len(daily_df):,}")
        metric_cols[1].metric(
            "Cumulative Time",
            self._format_duration(daily_df["estimated_total_time_sec"].sum()),
        )
        metric_cols[2].metric(
            "Average Daily Time",
            self._format_duration(daily_df["estimated_total_time_sec"].mean()),
        )
        metric_cols[3].metric("Latest Seen Day", daily_df.iloc[0]["date_label"])

        st.caption(f"Daily attendance_monitoring summary for {selected_name}.")
        records_container = st.container(border=True, height=520)
        with records_container:
            header_cols = st.columns([1.3, 2.4, 1.2, 1.2, 1.8, 1.1, 1.1, 2.2, 2.2])
            with header_cols[0]:
                st.markdown("**Date**")
            with header_cols[1]:
                st.markdown("**Daily Summary**")
            with header_cols[2]:
                st.markdown("**Detections**")
            with header_cols[3]:
                st.markdown("**Sessions**")
            with header_cols[4]:
                st.markdown("**Streams**")
            with header_cols[5]:
                st.markdown("**First Seen**")
            with header_cols[6]:
                st.markdown("**Last Seen**")
            with header_cols[7]:
                st.markdown("**Face Crop**")
            with header_cols[8]:
                st.markdown("**Screenshot**")

            for row in daily_df.itertuples(index=False):
                row_cols = st.columns(
                    [1.3, 2.4, 1.2, 1.2, 1.8, 1.1, 1.1, 2.2, 2.2],
                    vertical_alignment="center",
                )
                with row_cols[0]:
                    st.markdown(str(row.date_label))
                with row_cols[1]:
                    st.markdown(
                        f"**Time:** {row.estimated_total_time}<br>**Name:** {selected_name}",
                        unsafe_allow_html=True,
                    )
                with row_cols[2]:
                    st.markdown(f"{int(row.detections)}")
                with row_cols[3]:
                    st.markdown(f"{int(row.sessions)}")
                with row_cols[4]:
                    st.markdown(str(row.streams_display))
                with row_cols[5]:
                    st.markdown(str(row.first_seen_label))
                with row_cols[6]:
                    st.markdown(str(row.last_seen_label))
                with row_cols[7]:
                    _render_clickable_image_preview(
                        row.face_thumb,
                        _source_image_data_uri(row.face_image_path),
                        f"{selected_name} • {row.date_label} • Face Crop",
                        frame_height=190,
                        image_width=220,
                        image_height=160,
                    )
                with row_cols[8]:
                    _render_clickable_image_preview(
                        row.frame_thumb,
                        _source_image_data_uri(row.screenshot_path),
                        f"{selected_name} • {row.date_label} • Screenshot",
                        frame_height=190,
                        image_width=220,
                        image_height=160,
                    )
                st.divider()

    def render(self):
        st.title("Person Analytics")
        st.caption(
            "Person-wise attendance_monitoring insights from `/home/abdullah/PycharmProjects/Ai Classroom Attendance/outputs/database/activity_logs.db` → `attendance_table`."
        )

        detections_df = self._load_logs()
        if detections_df.empty:
            st.info("No attendance_monitoring detections are available in attendance_table.")
            return

        available_dates = sorted(detections_df["date_only"].dropna().unique().tolist())
        latest_date = max(available_dates)
        display_name_by_key = self._display_name_map(detections_df["username"])
        person_options = sorted(display_name_by_key.values(), key=lambda value: value.lower())
        stream_options = sorted(detections_df["stream_id"].dropna().unique().tolist())

        filter_cols = st.columns([1.6, 1.1, 1.1, 1.1, 1, 1])
        selected_name = filter_cols[0].selectbox(
            "Person",
            options=person_options,
            key="attendance_person_analytics_person",
        )
        selected_date = filter_cols[1].date_input(
            "Date",
            value=latest_date,
            min_value=min(available_dates),
            max_value=max(available_dates),
            key="attendance_person_analytics_date",
        )
        from_time = filter_cols[2].time_input(
            "From Time",
            value=dt_time(0, 0),
            key="attendance_person_analytics_from_time",
        )
        to_time = filter_cols[3].time_input(
            "To Time",
            value=dt_time(23, 59, 59),
            key="attendance_person_analytics_to_time",
        )
        selected_streams = filter_cols[4].multiselect(
            "Streams",
            options=stream_options,
            default=stream_options,
            key="attendance_person_analytics_streams",
        )
        gap_threshold_sec = filter_cols[5].number_input(
            "Session Gap (sec)",
            min_value=1,
            max_value=3600,
            value=60,
            step=1,
            key="attendance_person_analytics_gap",
        )

        start_ts = pd.Timestamp(datetime.combine(selected_date, from_time))
        end_ts = pd.Timestamp(datetime.combine(selected_date, to_time))
        if start_ts > end_ts:
            st.warning("From Time cannot be after To Time.")
            return

        day_df = detections_df[detections_df["date_only"] == selected_date].copy()
        if selected_streams:
            day_df = day_df[day_df["stream_id"].isin(selected_streams)].copy()
        day_df = day_df[
            (day_df["timestamp"] >= start_ts) & (day_df["timestamp"] <= end_ts)
        ].copy()

        sessions_df = self._build_sessions(day_df, int(gap_threshold_sec))

        selected_key = str(selected_name).strip().lower()
        person_df = day_df[
            day_df["username"].astype(str).str.strip().str.lower() == selected_key
        ].copy()
        person_sessions_df = sessions_df[
            sessions_df["person_key"] == selected_key
        ].copy() if not sessions_df.empty else pd.DataFrame()
        person_all_df = detections_df[
            detections_df["username"].astype(str).str.strip().str.lower() == selected_key
        ].copy()
        if selected_streams:
            person_all_df = person_all_df[
                person_all_df["stream_id"].isin(selected_streams)
            ].copy()
        person_all_sessions_df = self._build_sessions(
            person_all_df, int(gap_threshold_sec)
        )
        if not person_all_sessions_df.empty:
            person_all_sessions_df = person_all_sessions_df[
                person_all_sessions_df["person_key"] == selected_key
            ].copy()

        st.markdown(f"### {selected_name}")
        if person_df.empty:
            st.info("No detections are available for the selected person and filters.")
            return

        self._render_person_metrics(person_df, person_sessions_df)

        analytics_tabs = st.tabs(["Timeline", "Session Insights", "Daily Records"])
        with analytics_tabs[0]:
            self._render_timeline(person_sessions_df, selected_name)
        with analytics_tabs[1]:
            self._render_session_insights(person_sessions_df)
        with analytics_tabs[2]:
            self._render_daily_records(
                person_all_df,
                person_all_sessions_df,
                selected_name,
            )


class AttendanceApp:
    def __init__(self):
        config = get_application_config("attendance_monitoring")
        attendance_root = Path(config["root_dir"])
        attendance_root.mkdir(parents=True, exist_ok=True)

        table_names = config.get("table_names", {})
        attendance_table = table_names.get("attendance_monitoring", "attendance_table")
        live_frames_dir = Path(config.get("live_frames_dir", attendance_root / "live_frames"))
        stream_source_mode = config.get("stream_source_mode", "stream_api")
        streams_api_url = config.get("streams_api_url", "http://127.0.0.1:8080/api/streams")
        streams_public_base_url = config.get("streams_public_base_url", "")

        repo = SurveillanceRepository(
            face_db_path=config["face_db_path"],
            activity_db_path=config["activity_db_path"],
        )
        service = SurveillanceService(repo)

        self.pages = {
            "📷 Live Stream": AttendanceHomePage(
                attendance_root,
                live_frames_dir,
                stream_source_mode=stream_source_mode,
                streams_api_url=streams_api_url,
                streams_public_base_url=streams_public_base_url,
            ),
            "📊 Dashboard": AttendanceDashboardPage(
                attendance_root, Path(config["activity_db_path"]), attendance_table
            ),
            "🧠 Face Registration": FaceRegistrationPage(service),
            "📋 Attendance Logs": AttendanceLogsPage(
                attendance_root, Path(config["activity_db_path"]), attendance_table
            ),
            "👤 Person Analytics": PersonAnalyticsPage(
                attendance_root, Path(config["activity_db_path"]), attendance_table
            ),
        }

    def run(self):
        st.set_page_config(layout="wide")
        clear_persisted_theme_once()

        st.sidebar.title("Attendance Management")
        page = st.sidebar.selectbox("Select Page", list(self.pages.keys()))
        self.pages[page].render()
