from pathlib import Path
import base64
import html
import json
import sqlite3
import time
from datetime import datetime, time

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from lab_survelliance.utils import (
    FaceRegistrationPage,
    LiveCameraPage,
    SurveillanceRepository,
    SurveillanceService,
)
from utils.app_config import get_application_config
from utils.pagination import PaginationManager
from utils.theme_reset import clear_persisted_theme_once

CLASSROOM_CONFIG = get_application_config("classroom_surveillance")
ACTIVITY_IMAGE_DIR = Path(CLASSROOM_CONFIG.get("activity_images_dir", ""))
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _slugify_value(value: str) -> str:
    text = str(value).strip().lower()
    slug_chars = []
    prev_underscore = False
    for ch in text:
        if ch.isalnum():
            slug_chars.append(ch)
            prev_underscore = False
            continue
        if not prev_underscore:
            slug_chars.append("_")
            prev_underscore = True
    return "".join(slug_chars).strip("_")


@st.cache_data(ttl=5, show_spinner=False)
def _load_activity_image_index(directory: str):
    image_dir = Path(directory)
    if not image_dir.exists():
        return []

    entries = []
    for path in image_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        parts = path.stem.split("__")
        if len(parts) < 4 or not parts[0].startswith("act_"):
            continue

        uid = parts[0][4:].upper()
        activity_slug = parts[-2].lower()
        try:
            ts_ms = int(parts[-1])
        except ValueError:
            continue

        entries.append(
            {
                "uid": uid,
                "activity_slug": activity_slug,
                "ts_ms": ts_ms,
                "path": str(path),
            }
        )

    return entries


def _resolve_row_target_ms(row: pd.Series):
    for key in ("end_ts", "start_ts"):
        value = row.get(key)
        if pd.notna(value):
            try:
                return int(float(value) * 1000)
            except (TypeError, ValueError):
                pass

    end_dt = row.get("end_datetime")
    if pd.notna(end_dt):
        return int(pd.Timestamp(end_dt).timestamp() * 1000)
    return None


def _attach_activity_screenshot_paths(df: pd.DataFrame) -> pd.Series:
    index_entries = _load_activity_image_index(str(ACTIVITY_IMAGE_DIR))
    if not index_entries:
        return pd.Series([""] * len(df), index=df.index)

    by_uid_activity = {}
    by_uid = {}
    for entry in index_entries:
        uid = entry["uid"]
        activity_slug = entry["activity_slug"]
        ts_ms = entry["ts_ms"]
        path = entry["path"]

        by_uid_activity.setdefault((uid, activity_slug), []).append((ts_ms, path))
        by_uid.setdefault(uid, []).append((ts_ms, path))

    for values in by_uid_activity.values():
        values.sort(key=lambda item: item[0])
    for values in by_uid.values():
        values.sort(key=lambda item: item[0])

    def _pick_best(candidates, target_ms):
        if not candidates:
            return ""
        if target_ms is None:
            return candidates[-1][1]
        return min(candidates, key=lambda item: abs(item[0] - target_ms))[1]

    screenshot_paths = []
    for _, row in df.iterrows():
        uid = str(row.get("uid", "")).strip().upper()
        activity_slug = _slugify_value(row.get("activity", ""))
        target_ms = _resolve_row_target_ms(row)

        best_path = _pick_best(by_uid_activity.get((uid, activity_slug), []), target_ms)
        if not best_path:
            best_path = _pick_best(by_uid.get(uid, []), target_ms)

        if not best_path:
            screenshot_paths.append("")
            continue

        screenshot_paths.append(best_path)

    return pd.Series(screenshot_paths, index=df.index)


@st.cache_data(ttl=30, show_spinner=False)
def _thumbnail_data_uri(image_path: str, max_side: int = 140) -> str:
    if not image_path:
        return ""

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return ""

    image = cv2.imread(str(path))
    if image is None:
        return ""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 16
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        cropped = image[y0:y1, x0:x1]
        if cropped.size:
            image = cropped

    h, w = image.shape[:2]
    target = int(max_side)
    scale = float(target) / float(max(1, h))
    resized_w = max(1, int(round(w * scale)))
    resized = cv2.resize(image, (resized_w, target), interpolation=cv2.INTER_CUBIC)

    if resized_w >= target:
        x0 = (resized_w - target) // 2
        image = resized[:, x0 : x0 + target]
    else:
        canvas = np.full((target, target, 3), 10, dtype=np.uint8)
        x0 = (target - resized_w) // 2
        canvas[:, x0 : x0 + resized_w] = resized
        image = canvas

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        return ""

    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode(
        "ascii"
    )


@st.cache_data(ttl=30, show_spinner=False)
def _source_image_data_uri(image_path: str) -> str:
    if not image_path:
        return ""

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return ""

    try:
        image_bytes = path.read_bytes()
    except OSError:
        return ""

    if not image_bytes:
        return ""

    suffix = path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")
    return f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode("ascii")


def _render_clickable_image_preview(
    thumbnail_src: str,
    full_image_src: str,
    title: str,
    *,
    frame_height: int = 190,
    image_width: int = 220,
    image_height: int = 160,
):
    thumbnail_src = str(thumbnail_src or "").strip()
    full_image_src = str(full_image_src or thumbnail_src).strip() or thumbnail_src
    title_text = html.escape(str(title or "Image View"))

    if not thumbnail_src:
        components.html(
            f"""
            <style>
              body {{ margin: 0; background: transparent; }}
              .image-preview-empty {{
                width: {image_width}px;
                height: {image_height}px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 8px;
                border: 1px dashed rgba(148, 163, 184, 0.35);
                color: #9ca3af;
                font-size: 14px;
                background: rgba(2, 6, 23, 0.55);
                box-sizing: border-box;
              }}
            </style>
            <div class="image-preview-empty">No image</div>
            """,
            height=frame_height,
            scrolling=False,
        )
        return

    components.html(
        f"""
        <style>
          body {{
            margin: 0;
            background: transparent;
            font-family: "Trebuchet MS", "Verdana", sans-serif;
          }}
          .image-preview-button {{
            position: relative;
            display: inline-flex;
            padding: 0;
            border: none;
            background: transparent;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
          }}
          .image-preview-thumb {{
            width: {image_width}px;
            height: {image_height}px;
            object-fit: contain;
            background: rgba(2, 6, 23, 0.55);
            display: block;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.4);
          }}
          .image-preview-overlay {{
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(2, 6, 23, 0.68);
            color: #e5e7eb;
            font-size: 14px;
            font-weight: 700;
            opacity: 0;
            transition: opacity 0.18s ease;
          }}
          .image-preview-button:hover .image-preview-overlay {{
            opacity: 1;
          }}
          .image-preview-modal {{
            position: fixed;
            inset: 0;
            display: none;
            align-items: center;
            justify-content: center;
            background: rgba(2, 6, 23, 0.96);
            z-index: 9999;
            padding: 8px;
            box-sizing: border-box;
          }}
          .image-preview-modal.is-open {{
            display: flex;
          }}
          .image-preview-modal__content {{
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 14px;
          }}
          .image-preview-modal__topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
          }}
          .image-preview-modal__title {{
            color: #e5e7eb;
            font-size: 28px;
            font-weight: 700;
          }}
          .image-preview-modal__back {{
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(15, 23, 42, 0.88);
            color: #e5e7eb;
            border-radius: 10px;
            padding: 12px 20px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
          }}
          .image-preview-modal__viewport {{
            flex: 1;
            min-height: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: rgba(15, 23, 42, 0.45);
          }}
          .image-preview-modal__img {{
            max-width: 98vw;
            max-height: calc(100vh - 100px);
            object-fit: contain;
            display: block;
          }}
        </style>
        <button type="button" class="image-preview-button" onclick="openImagePreview()">
          <img src="{thumbnail_src}" alt="{title_text}" class="image-preview-thumb" />
          <span class="image-preview-overlay">Click for full view</span>
        </button>
        <div id="image-preview-modal" class="image-preview-modal" onclick="closeImagePreview(event)">
          <div class="image-preview-modal__content">
            <div class="image-preview-modal__topbar">
              <div class="image-preview-modal__title">{title_text}</div>
              <button type="button" class="image-preview-modal__back" onclick="closeImagePreview(event)">Back</button>
            </div>
            <div class="image-preview-modal__viewport">
              <img id="image-preview-full" class="image-preview-modal__img" src="{full_image_src}" alt="{title_text}" />
            </div>
          </div>
        </div>
        <script>
          function openImagePreview() {{
            var modal = document.getElementById("image-preview-modal");
            if (!modal) return;
            modal.classList.add("is-open");
            document.body.style.overflow = "hidden";
            if (modal.requestFullscreen) {{
              modal.requestFullscreen().catch(function() {{}});
            }}
          }}
          function closeImagePreview(event) {{
            if (event) {{
              event.preventDefault();
              event.stopPropagation();
            }}
            var modal = document.getElementById("image-preview-modal");
            if (!modal) return;
            modal.classList.remove("is-open");
            if (document.fullscreenElement) {{
              document.exitFullscreen().catch(function() {{}});
            }}
            document.body.style.overflow = "";
          }}
          document.addEventListener("keydown", function(event) {{
            if (event.key === "Escape") {{
              closeImagePreview();
            }}
          }});
        </script>
        """,
        height=frame_height,
        scrolling=False,
    )


class ClassroomLiveStreamPage(LiveCameraPage):
    LIVE_IMAGE_PATH = (
        Path(CLASSROOM_CONFIG["live_image_path"])
        if CLASSROOM_CONFIG.get("live_image_path")
        else None
    )
    LIVE_IMAGE_DIR = Path(CLASSROOM_CONFIG["live_frames_dir"])
    STATUS_DIR = Path(CLASSROOM_CONFIG["stream_status_dir"])
    STATUS_STALE_AFTER_SECONDS = 15
    REFRESH_INTERVAL_SECONDS = 3

    def __init__(self, service):
        super().__init__(service)
        self.title = "📷 Classroom Live Stream"

    def _load_stream_cards(self):
        if not self.STATUS_DIR.exists():
            return []

        cards = []
        for status_path in sorted(self.STATUS_DIR.glob("stream_*_status.json")):
            stream_slug = status_path.stem.replace("_status", "")
            frame_path = self.LIVE_IMAGE_DIR / f"{stream_slug}_latest_frame.jpg"
            payload = {}
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}

            last_update_text = str(payload.get("last_update") or "").strip()
            status_value = str(payload.get("status") or "").strip().lower()
            is_alive = False
            if last_update_text:
                try:
                    last_update = datetime.strptime(last_update_text, "%Y-%m-%d %H:%M:%S")
                    age_seconds = (datetime.now() - last_update).total_seconds()
                    is_alive = status_value == "running" and age_seconds <= self.STATUS_STALE_AFTER_SECONDS
                except ValueError:
                    is_alive = False

            cards.append(
                {
                    "stream_slug": stream_slug,
                    "stream_name": str(payload.get("display_name") or stream_slug.replace("_", " ").title()),
                    "frame_path": frame_path,
                    "payload": payload,
                    "alive": is_alive,
                }
            )

        return cards

    @st.fragment(run_every=REFRESH_INTERVAL_SECONDS)
    def _render_live_overview(self):
        stream_cards = self._load_stream_cards()
        # Filter to show only running/alive streams
        stream_cards = [card for card in stream_cards if card["alive"]]

        if not stream_cards:
            st.info(f"No live stream status files were found in: {self.STATUS_DIR}")
            return

        st.caption("Live classroom streams with their latest frame and runtime status.")

        cols = st.columns(2)
        for idx, card in enumerate(stream_cards):
            payload = card["payload"]
            with cols[idx % 2]:
                with st.container(border=True):
                    st.subheader(card["stream_name"])
                    st.caption(card["stream_slug"])

                    if card["alive"]:
                        st.success("Stream is running")
                    else:
                        st.error("Stream is stopped or stale")

                    if card["frame_path"].exists():
                        frame = self._load_frame(card["frame_path"])
                        if frame is None:
                            st.error(f"Failed to fetch image for {card['stream_slug']}.")
                        else:
                            try:
                                st.image(frame, use_container_width=True)
                            except OSError:
                                st.error(f"Failed to fetch image for {card['stream_slug']}.")
                    else:
                        st.info(f"No frame found for {card['stream_slug']}.")

                    metric_cols = st.columns(3)
                    metric_cols[0].metric(
                        "Frames",
                        int(payload.get("frame_count", 0) or 0),
                    )
                    metric_cols[1].metric(
                        "Detected",
                        int(payload.get("detection_count", 0) or 0),
                    )
                    metric_cols[2].metric(
                        "Recognized",
                        int(payload.get("recognized_count", 0) or 0),
                    )

                    st.caption(
                        f"Source: {payload.get('source', 'N/A')} | "
                        f"Last Update: {payload.get('last_update', 'N/A')} | "
                        f"PID: {payload.get('pid', 'N/A')}"
                    )

    def render(self):
        self.show_title()
        st.caption(
            f"Auto-refreshing every {self.REFRESH_INTERVAL_SECONDS} seconds when new frames or status updates arrive."
        )
        self._render_live_overview()


class ClassroomSurveillanceRepository(SurveillanceRepository):
    @staticmethod
    def _read_dataframe(db_path: str, query: str, params: tuple = None) -> pd.DataFrame:
        if not db_path:
            st.error("Database path is not configured properly.")
            return pd.DataFrame()
        
        db_path_obj = Path(db_path)
        if not db_path_obj.exists():
            st.error(f"Database file not found: {db_path}")
            return pd.DataFrame()
        
        try:
            with sqlite3.connect(str(db_path)) as conn:
                if params:
                    cursor = conn.execute(query, params)
                else:
                    cursor = conn.execute(query)
                rows = cursor.fetchall()
                columns = [column[0] for column in cursor.description]
            return pd.DataFrame(rows, columns=columns)
        except sqlite3.OperationalError as e:
            st.error(f"Database error: {e} (path: {db_path})")
            return pd.DataFrame()

    def get_logs(self, page: int = 1, per_page: int = 50, filters: dict = None):
        where_clauses = []
        params = []

        if filters:
            if filters.get("stream_name"):
                where_clauses.append("stream_name = ?")
                params.append(filters["stream_name"])
            if filters.get("student_name"):
                where_clauses.append("student_name = ?")
                params.append(filters["student_name"])
            if filters.get("start_date"):
                where_clauses.append("unix_ts >= ?")
                params.append(filters["start_date"])
            if filters.get("end_date"):
                where_clauses.append("unix_ts <= ?")
                params.append(filters["end_date"])
            if filters.get("activity"):
                where_clauses.append("activity = ?")
                params.append(filters["activity"])
            if filters.get("attentive") is not None:
                where_clauses.append("attentive = ?")
                params.append(int(filters["attentive"]))

        query = "SELECT * FROM attention_logs"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY unix_ts DESC, id DESC"
        query += " LIMIT ? OFFSET ?"

        offset = (page - 1) * per_page
        params.extend([per_page, offset])

        df = self._read_dataframe(self.activity_db, query, tuple(params))

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def get_logs_count(self, filters: dict = None):
        where_clauses = []
        params = []

        if filters:
            if filters.get("stream_name"):
                where_clauses.append("stream_name = ?")
                params.append(filters["stream_name"])
            if filters.get("student_name"):
                where_clauses.append("student_name = ?")
                params.append(filters["student_name"])
            if filters.get("start_date"):
                where_clauses.append("unix_ts >= ?")
                params.append(filters["start_date"])
            if filters.get("end_date"):
                where_clauses.append("unix_ts <= ?")
                params.append(filters["end_date"])
            if filters.get("activity"):
                where_clauses.append("activity = ?")
                params.append(filters["activity"])
            if filters.get("attentive") is not None:
                where_clauses.append("attentive = ?")
                params.append(int(filters["attentive"]))

        query = "SELECT COUNT(*) FROM attention_logs"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        if not self.activity_db:
            return 0
        
        try:
            with sqlite3.connect(self.activity_db) as conn:
                cursor = conn.execute(query, tuple(params))
                result = cursor.fetchone()
                return result[0] if result else 0
        except sqlite3.OperationalError:
            return 0

    def get_attention_summary(self, page: int = 1, per_page: int = 50, filters: dict = None):
        where_clauses = []
        params = []

        if filters:
            if filters.get("stream_name"):
                where_clauses.append("stream_name = ?")
                params.append(filters["stream_name"])
            if filters.get("student_name"):
                where_clauses.append("student_name = ?")
                params.append(filters["student_name"])

        query = "SELECT * FROM student_attention_summary"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY total_seen_seconds DESC"
        query += " LIMIT ? OFFSET ?"

        offset = (page - 1) * per_page
        params.extend([per_page, offset])

        df = self._read_dataframe(self.activity_db, query, tuple(params))

        if df.empty:
            return df

        for col in ("first_seen", "last_seen"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def get_attention_summary_count(self, filters: dict = None):
        where_clauses = []
        params = []

        if filters:
            if filters.get("stream_name"):
                where_clauses.append("stream_name = ?")
                params.append(filters["stream_name"])
            if filters.get("student_name"):
                where_clauses.append("student_name = ?")
                params.append(filters["student_name"])

        query = "SELECT COUNT(*) FROM student_attention_summary"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        if not self.activity_db:
            return 0
        
        try:
            with sqlite3.connect(self.activity_db) as conn:
                cursor = conn.execute(query, tuple(params))
                result = cursor.fetchone()
                return result[0] if result else 0
        except sqlite3.OperationalError:
            return 0


class ClassroomSurveillanceService(SurveillanceService):
    def get_logs(self, page: int = 1, per_page: int = 50, filters: dict = None):
        return self.repo.get_logs(page=page, per_page=per_page, filters=filters)

    def get_logs_count(self, filters: dict = None):
        return self.repo.get_logs_count(filters=filters)

    def get_attention_summary(self, page: int = 1, per_page: int = 50, filters: dict = None):
        return self.repo.get_attention_summary(page=page, per_page=per_page, filters=filters)

    def get_attention_summary_count(self, filters: dict = None):
        return self.repo.get_attention_summary_count(filters=filters)


def _classroom_student_label(name, track_id):
    clean_name = str(name or "").strip()
    if clean_name and clean_name.lower() != "unknown":
        return clean_name
    return f"Unknown (Track {track_id})"


def _classroom_activity_label(activity, attentive):
    clean_activity = str(activity or "").strip()
    if clean_activity and clean_activity.lower() != "none":
        return clean_activity
    return "Attentive" if int(bool(attentive)) else "Inattentive"


def _classroom_prepare_logs(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty:
        return logs_df.copy()

    df = logs_df.copy()
    df["student_label"] = [
        _classroom_student_label(name, track_id)
        for name, track_id in zip(df["student_name"], df["track_id"])
    ]
    df["activity_label"] = [
        _classroom_activity_label(activity, attentive)
        for activity, attentive in zip(df["activity"], df["attentive"])
    ]
    df["attention_state"] = df["attentive"].apply(
        lambda value: "Attentive" if int(bool(value)) else "Inattentive"
    )
    df["stream_label"] = (
        df["stream_display_name"].fillna("").astype(str).str.strip().replace("", pd.NA)
    )
    df["stream_label"] = df["stream_label"].fillna(df["stream_name"])
    df["head_direction"] = (
        df["head_direction"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )
    df["time_bucket"] = df["timestamp"].dt.floor("min")
    return df


def _classroom_prepare_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    df = summary_df.copy()
    df["student_label"] = (
        df["display_name"].fillna("").astype(str).str.strip().replace("", pd.NA)
    )
    df["student_label"] = df["student_label"].fillna(df["student_key"])
    df["stream_label"] = df["stream_name"].fillna("Unknown").astype(str)
    df["attention_state"] = df["last_attention_state"].apply(
        lambda value: "Attentive" if int(bool(value)) else "Inattentive"
    )
    return df


def _classroom_apply_dashboard_filters(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty:
        return logs_df.copy()

    min_ts = logs_df["timestamp"].min()
    max_ts = logs_df["timestamp"].max()
    default_start = min_ts.date() if pd.notna(min_ts) else datetime.now().date()
    default_end = max_ts.date() if pd.notna(max_ts) else datetime.now().date()
    default_start_time = min_ts.time().replace(microsecond=0) if pd.notna(min_ts) else time(0, 0)
    default_end_time = max_ts.time().replace(microsecond=0) if pd.notna(max_ts) else time(23, 59)

    st.caption("Filter the classroom analytics by date and time here. Use the sidebar for stream, attention state, and activity filters.")

    st.markdown(
        """
        <style>
          div[data-testid="stMultiSelect"] [data-baseweb="value-container"],
          div[data-testid="stMultiSelect"] [data-baseweb="tags"] {
            flex-wrap: nowrap !important;
            white-space: nowrap !important;
            overflow: hidden !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    stream_options = sorted(logs_df["stream_label"].dropna().unique())
    state_options = sorted(logs_df["attention_state"].dropna().unique())
    activity_options = sorted(logs_df["activity_label"].dropna().unique())

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(
        [1.15, 1.15, 1.05, 1.05]
    )
    start_date = filter_col1.date_input("Start Date", value=default_start, key="classroom_dash_start_date")
    end_date = filter_col2.date_input("End Date", value=default_end, key="classroom_dash_end_date")
    start_time = filter_col3.time_input("Start Time", value=default_start_time, key="classroom_dash_start_time")
    end_time = filter_col4.time_input("End Time", value=default_end_time, key="classroom_dash_end_time")

    with st.sidebar:
        st.markdown("### Dashboard Filters")
        selected_streams = st.multiselect(
            "Streams",
            options=stream_options,
            default=stream_options,
            key="classroom_dash_streams",
        )
        selected_states = st.multiselect(
            "Attention States",
            options=state_options,
            default=state_options,
            key="classroom_dash_states",
        )
        selected_activities = st.multiselect(
            "Activities",
            options=activity_options,
            default=activity_options,
            key="classroom_dash_activities",
        )

    start_dt = pd.Timestamp.combine(start_date, start_time)
    end_dt = pd.Timestamp.combine(end_date, end_time)
    if end_dt < start_dt:
        st.warning("End date/time is earlier than start date/time. Showing no records.")
        return logs_df.iloc[0:0].copy()

    filtered_df = logs_df[
        (logs_df["timestamp"] >= start_dt)
        & (logs_df["timestamp"] <= end_dt)
        & (logs_df["stream_label"].isin(selected_streams))
        & (logs_df["attention_state"].isin(selected_states))
        & (logs_df["activity_label"].isin(selected_activities))
    ].copy()

    return filtered_df


def _classroom_summary_from_logs(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty:
        return pd.DataFrame()

    summary_df = (
        logs_df.sort_values(["student_label", "timestamp"])
        .groupby(["student_label", "stream_label"], as_index=False)
        .agg(
            total_seen_seconds=("total_seen_seconds", "max"),
            attentive_seconds=("attentive_seconds", "max"),
            inattentive_seconds=("inattentive_seconds", "max"),
            attention_ratio=("attention_ratio", "max"),
            last_attention_state=("attention_state", "last"),
            last_activity=("activity_label", "last"),
            last_head_direction=("head_direction", "last"),
            last_seen=("timestamp", "max"),
        )
    )
    summary_df["attention_state"] = summary_df["last_attention_state"]
    return summary_df


class ClassroomActivitiesDashboardPage:
    def __init__(self, service):
        self.title = "📊 Classroom Dashboard Graphs"
        self.service = service

    def show_title(self):
        st.subheader(self.title)

    def render(self):
        self.show_title()
        faces_df, logs_df = self.service.load_data()

        logs_df = _classroom_prepare_logs(logs_df)
        if logs_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Registered Faces", len(faces_df))
            col2.metric("Students Seen", 0)
            col3.metric("Attention Ratio", "0%")
            col4.metric("Tracked Time", "0 min")
            st.info("No classroom activity logs are available yet.")
            return

        logs_df = _classroom_apply_dashboard_filters(logs_df)
        summary_df = _classroom_summary_from_logs(logs_df)

        if logs_df.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Attention Ratio", "0%")
            col2.metric("Tracked Time", "0 min")
            col3.metric("Streams", 0)
            col4.metric("Activity Types", 0)
            col5.metric("Log Records", 0)
            st.info("No classroom activity logs match the selected filters.")
            return

        col1, col2, col3, col4, col5 = st.columns(5)
        attention_ratio = logs_df["attentive"].mean() * 100 if not logs_df.empty else 0.0
        tracked_minutes = logs_df["total_seen_seconds"].max() / 60.0 if "total_seen_seconds" in logs_df else 0.0
        col1.metric("Attention Ratio", f"{attention_ratio:.1f}%")
        col2.metric("Tracked Time", f"{tracked_minutes:.1f} min")
        col3.metric("Streams", logs_df["stream_label"].nunique())
        col4.metric("Activity Types", logs_df["activity_label"].nunique())
        col5.metric("Log Records", len(logs_df))

        st.divider()

        flow_df = (
            logs_df.groupby("time_bucket", as_index=False)
            .agg(
                attentive_students=("attentive", "sum"),
                total_students=("attentive", "size"),
            )
            .sort_values("time_bucket")
        )
        flow_df["attention_rate"] = (
            flow_df["attentive_students"] / flow_df["total_students"]
        ).fillna(0.0) * 100

        activity_dist = (
            logs_df.groupby("activity_label", as_index=False)
            .agg(records=("id", "count"))
            .sort_values("records", ascending=False)
        )

        flow_fig = px.line(
            flow_df,
            x="time_bucket",
            y="attention_rate",
            markers=True,
            title="Attention Flow Over Time",
        )
        flow_fig.update_traces(
            line=dict(width=3, color="#0f766e"),
            marker=dict(size=7, color="#f59e0b"),
            hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Attention: %{y:.1f}%<extra></extra>",
        )
        flow_fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Attention Rate (%)",
            yaxis_range=[0, 100],
            margin=dict(l=10, r=10, t=50, b=10),
        )

        donut_fig = px.pie(
            activity_dist,
            names="activity_label",
            values="records",
            hole=0.58,
            title="Activity Distribution",
        )
        donut_fig.update_traces(
            textinfo="percent+label",
            hovertemplate="Activity: %{label}<br>Records: %{value}<extra></extra>",
        )
        donut_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))

        if summary_df.empty:
            st.info("Student summary data is not available for the selected filters.")
            return

        top_students_df = summary_df.sort_values("total_seen_seconds", ascending=False).head(10).copy()
        top_students_df["total_minutes"] = top_students_df["total_seen_seconds"] / 60.0
        top_students_df["attentive_minutes"] = top_students_df["attentive_seconds"] / 60.0
        top_students_df["inattentive_minutes"] = top_students_df["inattentive_seconds"] / 60.0

        state_dist = (
            summary_df.groupby("attention_state", as_index=False)
            .agg(students=("student_label", "nunique"))
        )
        head_dist = (
            logs_df.groupby("head_direction", as_index=False)
            .agg(records=("id", "count"))
            .sort_values("records", ascending=False)
        )
        stream_activity_df = (
            logs_df.groupby(["stream_label", "activity_label"], as_index=False)
            .agg(records=("id", "count"))
        )

        student_fig = px.bar(
            top_students_df,
            x="student_label",
            y=["attentive_minutes", "inattentive_minutes"],
            title="Student Attention Time",
            barmode="stack",
        )
        student_fig.update_layout(
            xaxis_title="Student",
            yaxis_title="Minutes",
            legend_title_text="Time Type",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        student_fig.update_traces(
            hovertemplate="Student: %{x}<br>Minutes: %{y:.1f}<extra></extra>"
        )

        state_fig = px.pie(
            state_dist,
            names="attention_state",
            values="students",
            hole=0.5,
            title="Current Student Attention Split",
            color="attention_state",
            color_discrete_map={"Attentive": "#10b981", "Inattentive": "#ef4444"},
        )
        state_fig.update_traces(
            textinfo="percent+label",
            hovertemplate="State: %{label}<br>Students: %{value}<extra></extra>",
        )
        state_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))

        head_fig = px.bar(
            head_dist,
            x="head_direction",
            y="records",
            color="head_direction",
            title="Head Direction Distribution",
        )
        head_fig.update_layout(
            xaxis_title="Head Direction",
            yaxis_title="Log Records",
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
        )

        stream_fig = px.bar(
            stream_activity_df,
            x="stream_label",
            y="records",
            color="activity_label",
            title="Stream Activity Mix",
        )
        stream_fig.update_layout(
            xaxis_title="Stream",
            yaxis_title="Log Records",
            legend_title_text="Activity",
            barmode="stack",
            margin=dict(l=10, r=10, t=50, b=10),
        )

        overview_charts = {
            "Attention Flow": flow_fig,
            "Activity Distribution": donut_fig,
            "Attention Split": state_fig,
        }
        breakdown_charts = {
            "Student Attention Time": student_fig,
            "Head Direction": head_fig,
            "Stream Activity Mix": stream_fig,
        }

        selected_category = st.segmented_control(
            "Graph Category",
            options=["Overview Graphs", "Breakdown Graphs"],
            default="Overview Graphs",
            key="classroom_dashboard_category_toggle",
        )

        if selected_category == "Overview Graphs":
            st.subheader("Overview Graphs")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Attention Flow")
                st.plotly_chart(flow_fig, use_container_width=True)
            with col2:
                st.markdown("#### Activity Distribution")
                st.plotly_chart(donut_fig, use_container_width=True)
            with col3:
                st.markdown("#### Attention Split")
                st.plotly_chart(state_fig, use_container_width=True)
        else:
            st.subheader("Breakdown Graphs")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Student Attention Time")
                st.plotly_chart(student_fig, use_container_width=True)
            with col2:
                st.markdown("#### Head Direction")
                st.plotly_chart(head_fig, use_container_width=True)
            with col3:
                st.markdown("#### Stream Activity Mix")
                st.plotly_chart(stream_fig, use_container_width=True)




class ClassroomStudentsLogsPage:
    def __init__(self, service):
        self.title = "📋 Students Activity Logs"
        self.service = service

    def show_title(self):
        st.subheader(self.title)

    def _render_student_timeline(self, student_df, student_name):
        """Render timeline graph for a specific student"""
        if student_df.empty:
            st.info(f"No timeline data available for {student_name}.")
            return

        timeline_df = student_df.copy().sort_values("timestamp", ascending=True)
        
        st.markdown(f"#### 📈 Activity Analytics - {student_name}")
        
        graph_toggle = st.segmented_control(
            "Select Graph",
            options=["Timeline", "Activity Distribution", "Attention Split"],
            default="Timeline",
            key=f"student_graph_toggle_{student_name}",
        )
        
        timeline_fig = px.scatter(
            timeline_df, x="timestamp", y="activity_label", color="attention_state",
            size="attention_ratio", size_max=15,
            color_discrete_map={"Attentive": "#10b981", "Inattentive": "#ef4444"},
            hover_data=["stream_label", "head_direction", "attention_ratio"],
        )
        timeline_fig.update_layout(xaxis_title="Time", yaxis_title="Activity", height=400, margin=dict(l=10, r=10, t=50, b=10))

        activity_dist = timeline_df.groupby("activity_label", as_index=False).agg(records=("id", "count")).sort_values("records", ascending=False)
        activity_fig = px.bar(activity_dist, x="activity_label", y="records", color="activity_label", title="Activity Breakdown")
        activity_fig.update_layout(xaxis_title="Activity", yaxis_title="Records", showlegend=False, margin=dict(l=10, r=10, t=50, b=10))

        state_dist = timeline_df.groupby("attention_state", as_index=False).agg(records=("id", "count"))
        state_fig = px.pie(state_dist, names="attention_state", values="records", title="Attention State Split",
                          color="attention_state", color_discrete_map={"Attentive": "#10b981", "Inattentive": "#ef4444"})
        state_fig.update_traces(textinfo="percent+label", hole=0.4)

        if graph_toggle == "Timeline":
            st.plotly_chart(timeline_fig, use_container_width=True)
        elif graph_toggle == "Activity Distribution":
            st.plotly_chart(activity_fig, use_container_width=True)
        elif graph_toggle == "Attention Split":
            st.plotly_chart(state_fig, use_container_width=True)

    def _render_activity_table(self, logs_df):
        """Render the activity logs table with screenshots"""
        if logs_df.empty:
            st.info("No records match the selected filters.")
            return

        st.caption(f"Showing {len(logs_df)} record(s).")

        rows_html = []
        for _, row in logs_df.iterrows():
            timestamp = pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            stream = html.escape(str(row["stream_label"]))
            student = html.escape(str(row["student_label"]))
            attention_state = html.escape(str(row["attention_state"]))
            activity = html.escape(str(row["activity_label"]))
            head_direction = html.escape(str(row["head_direction"]))
            attention_ratio = f"{float(row['attention_ratio']) * 100:.1f}%"
            thumb_src = str(row.get("screenshot_thumb") or "").strip()
            image_cell = f'<img src="{thumb_src}" class="shot-thumb" alt="Screenshot" />' if thumb_src else '<div class="shot-empty">No image</div>'
            rows_html.append(f"<tr><td>{timestamp}</td><td>{stream}</td><td>{student}</td><td>{attention_state}</td><td>{activity}</td><td>{head_direction}</td><td>{attention_ratio}</td><td>{image_cell}</td></tr>")

        table_html = f"""
        <style>
          .student-logs-wrap {{ max-height: 760px; overflow-y: auto; border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 12px; }}
          .student-logs-table {{ width: 100%; border-collapse: collapse; font-family: "Trebuchet MS", sans-serif; font-size: 16px; }}
          .student-logs-table thead th {{ position: sticky; top: 0; z-index: 2; background: #111827; color: #e5e7eb; text-align: left; padding: 14px 12px; border-bottom: 1px solid rgba(148, 163, 184, 0.35); }}
          .student-logs-table td {{ color: #d1d5db; padding: 14px 12px; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }}
          .student-logs-table tbody tr:hover {{ background: rgba(30, 41, 59, 0.45); }}
          .shot-thumb {{ width: 220px; height: 160px; object-fit: contain; background: rgba(2, 6, 23, 0.55); display: block; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.4); }}
          .shot-empty {{ width: 220px; height: 160px; display: flex; align-items: center; justify-content: center; border-radius: 8px; border: 1px dashed rgba(148, 163, 184, 0.35); color: #9ca3af; font-size: 14px; }}
        </style>
        <div class="student-logs-wrap"><table class="student-logs-table">
          <thead><tr><th>Timestamp</th><th>Stream</th><th>Student</th><th>Attention State</th><th>Activity</th><th>Head Direction</th><th>Attention Ratio</th><th>Screenshot</th></tr></thead>
          <tbody>{"".join(rows_html)}</tbody>
        </table></div>"""
        components.html(table_html, height=850)
        st.download_button("⬇ Download CSV", logs_df.to_csv(index=False).encode("utf-8"), "classroom_student_logs.csv", key="classroom_logs_download")

    def render(self):
        self.show_title()
        
        # Get total count for pagination
        total_count = self.service.get_logs_count()
        
        # Create paginator
        paginator = PaginationManager('classroom_logs', total_count, default_per_page=50)
        
        # Load paginated data
        logs_df = _classroom_prepare_logs(self.service.get_logs(page=paginator.current_page, per_page=paginator.per_page))

        if logs_df.empty:
            st.info("No classroom student logs are available yet.")
            return

        logs_df = logs_df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        logs_df["screenshot_path"] = _attach_activity_screenshot_paths(logs_df)
        logs_df["screenshot_thumb"] = logs_df["screenshot_path"].apply(lambda p: _thumbnail_data_uri(p, max_side=300))

        stream_options = ["All Streams"] + sorted(logs_df["stream_label"].dropna().unique())
        state_options = ["All States", "Attentive", "Inattentive"]
        student_options = ["All Students"] + sorted(logs_df["student_label"].dropna().unique())

        # All filters in a single row
        f1, f2, f3, f4, f5, f6, f7, f8 = st.columns(8)
        with f1:
            selected_stream = st.selectbox("Stream", stream_options)
        with f2:
            selected_state = st.selectbox("State", state_options)
        with f3:
            selected_student = st.selectbox("Student", student_options)
        
        min_date = logs_df["timestamp"].min().date()
        max_date = logs_df["timestamp"].max().date()
        with f4:
            start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="table_filter_start_date")
        with f5:
            end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date, key="table_filter_end_date")
        
        activity_options = ["All Activities"] + sorted(logs_df["activity_label"].dropna().unique())
        with f6:
            selected_activity = st.selectbox("Activity", activity_options, key="table_filter_activity")
        
        head_options = ["All Directions"] + sorted(logs_df["head_direction"].dropna().unique())
        with f7:
            selected_head = st.selectbox("Head", head_options, key="table_filter_head")
        
        with f8:
            min_attention = st.number_input("Min Att. %", min_value=0.0, max_value=100.0, value=0.0, step=10.0, key="table_filter_min_attention")

        filtered_df = logs_df.copy()
        if selected_stream != "All Streams":
            filtered_df = filtered_df[filtered_df["stream_label"] == selected_stream]
        if selected_state != "All States":
            filtered_df = filtered_df[filtered_df["attention_state"] == selected_state]
        if selected_student != "All Students":
            filtered_df = filtered_df[filtered_df["student_label"] == selected_student]

        if filtered_df.empty:
            st.info("No records match the selected filters.")
            return

        # Apply additional filters
        final_filtered_df = filtered_df.copy()
        start_dt = pd.Timestamp.combine(start_date, pd.Timestamp.min.time())
        end_dt = pd.Timestamp.combine(end_date, pd.Timestamp.max.time())
        final_filtered_df = final_filtered_df[(final_filtered_df["timestamp"] >= start_dt) & (final_filtered_df["timestamp"] <= end_dt)]
        
        if selected_activity != "All Activities":
            final_filtered_df = final_filtered_df[final_filtered_df["activity_label"] == selected_activity]
        if selected_head != "All Directions":
            final_filtered_df = final_filtered_df[final_filtered_df["head_direction"] == selected_head]
        final_filtered_df = final_filtered_df[(final_filtered_df["attention_ratio"] * 100 >= min_attention)]
        
        if final_filtered_df.empty:
            st.info("No records match all the selected filters.")
            tab1, tab2 = st.tabs(["📈 Student Timeline & Analytics", "📋 Activity Logs Table"])
            with tab1, tab2:
                st.info("No data to display.")
            return

        st.caption(f"Showing {len(final_filtered_df)} record(s) after all filters.")

        tab1, tab2 = st.tabs(["📈 Student Timeline & Analytics", "📋 Activity Logs Table"])
        with tab1:
            st.markdown("### Student Activity Timeline & Analytics")
            self._render_student_timeline(final_filtered_df, selected_student)
        with tab2:
            st.markdown("### Detailed Activity Records")
            self._render_activity_table(final_filtered_df)
        
        # Pagination controls AFTER the table
        if paginator.render_pagination_controls():
            st.rerun()


class ClassroomSurveillanceApp:
    def __init__(self):
        repo = ClassroomSurveillanceRepository(face_db_path=CLASSROOM_CONFIG["face_db_path"], activity_db_path=CLASSROOM_CONFIG["student_logs_db_path"])
        service = ClassroomSurveillanceService(repo)
        self.pages = {
            "📷 Live Stream": ClassroomLiveStreamPage(service),
            "🧠 Face Registration": FaceRegistrationPage(service),
            "📊 Dashboard Graphs of Activities": ClassroomActivitiesDashboardPage(service),
            "📋 Students Activity Logs": ClassroomStudentsLogsPage(service),
        }

    def run(self):
        st.set_page_config(layout="wide")
        clear_persisted_theme_once()
        st.sidebar.title("Classroom Surveillance")
        page = st.sidebar.selectbox("Select Page", list(self.pages.keys()))
        self.pages[page].render()
