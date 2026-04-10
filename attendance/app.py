import streamlit as st
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

    def __init__(self, attendance_root: Path):
        self.attendance_root = attendance_root
        self.live_frames_root = attendance_root / "live_frames"

    def _load_stream_cards(self):
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

            alive = False
            if isinstance(payload, dict):
                last_update_text = str(payload.get("last_update") or "").strip()
                status_value = str(payload.get("status") or "").strip().lower()
                if last_update_text:
                    try:
                        last_update = datetime.strptime(
                            last_update_text, "%Y-%m-%d %H:%M:%S"
                        )
                        age_seconds = (datetime.now() - last_update).total_seconds()
                        alive = status_value == "running" and age_seconds <= self.STATUS_STALE_AFTER_SECONDS
                    except ValueError:
                        alive = False

            cards.append(
                {
                    "stream_name": stream_slug.replace("_", " ").title(),
                    "status_path": status_path,
                    "frame_path": frame_path,
                    "payload": payload if isinstance(payload, dict) else {},
                    "alive": alive,
                }
            )

        return cards

    @st.fragment(run_every=5)
    def render(self):
        st.title("Walk Through Attendance System")
        st.caption("Live attendance stream overview")

        stream_cards = self._load_stream_cards()
        # Filter to show only running/alive streams
        stream_cards = [card for card in stream_cards if card["alive"]]
        if not stream_cards:
            st.info("No live attendance streams were found.")
            return

        cols = st.columns(2)
        for idx, card in enumerate(stream_cards):
            payload = card["payload"]
            with cols[idx % 2]:
                with st.container(border=True):
                    st.subheader(card["stream_name"])
                    if card["alive"]:
                        st.success("stream is alive")
                    else:
                        st.error("stream is not alive")

                    if card["frame_path"].exists():
                        st.image(str(card["frame_path"]), use_container_width=True)
                    else:
                        st.info("No live frame available.")

                    if card["alive"]:
                        metric_cols = st.columns(3)
                        metric_cols[0].metric(
                            "Detected", int(payload.get("detected_count", 0) or 0)
                        )
                        metric_cols[1].metric(
                            "Recognized", int(payload.get("recognized_count", 0) or 0)
                        )
                        metric_cols[2].metric(
                            "FPS", f"{float(payload.get('fps', 0.0) or 0.0):.2f}"
                        )

                        st.caption(
                            f"Source: {payload.get('source', 'N/A')} | "
                            f"Started: {payload.get('stream_started_at', 'N/A')} | "
                            f"Last Update: {payload.get('last_update', 'N/A')}"
                        )
                    else:
                        st.caption(
                            "Status JSON is missing, invalid, not running, or not updating."
                        )


class AttendanceLogsPage:
    def __init__(self, attendance_root: Path):
        self.attendance_root = attendance_root
        self.db_path = attendance_root / "faces.db"

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
                FROM attendance_table
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
                f"SELECT COUNT(*) FROM attendance_table {where_sql}",
                params,
            )
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
        finally:
            conn.close()

    def render(self):
        st.title("Attendance Logs")
        st.caption("Search and review saved attendance records.")

        df = self._load_logs()
        if df.empty:
            st.info("No attendance logs are available in faces.db.")
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
            st.info("No attendance logs match the selected filters.")
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

        st.caption(f"Showing {page_df.shape[0]} of {total_count} attendance record(s).")

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
    def __init__(self, attendance_root: Path):
        self.attendance_root = attendance_root
        self.db_path = attendance_root / "faces.db"

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
                FROM attendance_table
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
                f"SELECT COUNT(*) FROM attendance_table {where_sql}",
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
        st.caption("Analytics and insights for attendance data")

        df = self._load_logs()
        if df.empty:
            st.info("No attendance logs are available in faces.db.")
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


class AttendanceApp:
    def __init__(self):
        config = get_application_config("attendance")
        attendance_root = Path(config["root_dir"])
        attendance_root.mkdir(parents=True, exist_ok=True)

        repo = SurveillanceRepository(
            face_db_path=config["face_db_path"],
            activity_db_path=config["activity_db_path"],
        )
        service = SurveillanceService(repo)

        self.pages = {
            "🏠 Home": AttendanceHomePage(attendance_root),
            "📊 Dashboard": AttendanceDashboardPage(attendance_root),
            "🧠 Face Registration": FaceRegistrationPage(service),
            "📋 Attendance Logs": AttendanceLogsPage(attendance_root),
        }

    def run(self):
        st.set_page_config(layout="wide")
        clear_persisted_theme_once()

        st.sidebar.title("Walk Through Attendance")
        page = st.sidebar.selectbox("Select Page", list(self.pages.keys()))
        self.pages[page].render()
