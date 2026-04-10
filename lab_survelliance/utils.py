# utils.py

import sqlite3
import time
import base64
import html
import json
import threading
from collections import Counter
from collections import deque
from datetime import datetime, time as dt_time
from pathlib import Path
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover - optional dependency
    st_autorefresh = None

try:
    import av
except Exception:  # pragma: no cover - optional dependency
    av = None
try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional dependency
    mp = None
try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
except Exception:  # pragma: no cover - optional dependency
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None

from utils.app_config import get_application_config, load_app_config
from utils.pagination import PaginationManager

LAB_CONFIG = get_application_config("lab_surveillance")
CVML_SURVEILLANCE_ROOT = Path(LAB_CONFIG["root_dir"])
ACTIVITY_IMAGE_DIR = Path(LAB_CONFIG["activity_images_dir"])
REGION_OUTPUT_DIR = Path(LAB_CONFIG["region_output_dir"])
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_REGISTRATION_DEFAULTS = {
    "guide_width_ratio": 0.38,
    "guide_height_ratio": 0.58,
    "guide_min_size_ratio": 0.08,
    "guide_max_size_ratio": 0.92,
    "raw_yaw_threshold": 30.0,
    "raw_pitch_up_threshold": -14.0,
    "raw_pitch_down_threshold": 18.0,
    "raw_pitch_deadzone": 10.0,
    "frontal_yaw_deadzone": 33.0,
    "frontal_pitch_deadzone": 10.0,
    "baseline_left_threshold": -42.0,
    "baseline_right_threshold": 12.0,
    "baseline_up_pitch_threshold": 5.0,
    "baseline_down_pitch_threshold": -4.0,
    "baseline_up_vertical_ratio_delta": -0.12,
    "baseline_down_vertical_ratio_delta": 0.10,
    "baseline_level_vertical_ratio_deadzone": 0.06,
    "fallback_left_yaw_ratio": -0.20,
    "fallback_right_yaw_ratio": 0.20,
    "fallback_up_pitch_ratio": 0.15,
    "fallback_down_pitch_ratio": 0.03,
    "face_crop_expand_ratio": 0.18,
    "pose_vote_window": 3,
    "angle_smooth_factor": 0.75,
}


def _is_secure_camera_context() -> bool:
    url = str(getattr(st.context, "url", "") or "").strip()
    if not url:
        return True

    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower()
    if parsed.scheme == "https":
        return True
    return host in {"localhost", "127.0.0.1", "::1"}


def _face_registration_settings() -> dict:
    config = load_app_config().get("face_registration", {})
    if not isinstance(config, dict):
        return dict(FACE_REGISTRATION_DEFAULTS)
    settings = dict(FACE_REGISTRATION_DEFAULTS)
    settings.update(config)
    return settings


def _exclude_analyzing(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["activity"].astype(str).str.upper() != "ANALYZING"].copy()


def _canonical_activity_label(value: str):
    text = str(value or "").strip().lower()
    if not text or "analyzing" in text:
        return None
    if "mobile" in text or "phone" in text:
        return "Using Mobile"
    if "sleep" in text:
        return "Sleeping"
    if "head down" in text or "head_down" in text or ("head" in text and "down" in text):
        return "Head Down"
    if "not working" in text:
        return "Not Working"
    if "working" in text:
        return "Working"
    return None


def _activity_filter_labels(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    if "analyzing" in text:
        return []

    labels = []
    if "mobile" in text or "phone" in text:
        labels.append("Using Mobile")
    if "sleep" in text:
        labels.append("Sleeping")
    if "head down" in text or "head_down" in text or ("head" in text and "down" in text):
        labels.append("Head Down")
    if "not working" in text or "|no" in text:
        labels.append("Not Working")
    if "working" in text and "Not Working" not in labels:
        labels.append("Working")
    if "seated" in text or "sitting" in text:
        labels.append("Seated")
    if "standing" in text:
        labels.append("Standing")
    return labels


def _format_person_status_activity(value: str) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    if not text:
        return ""

    parts = [part.strip() for part in text.split("|") if part.strip()]
    normalized_parts = []

    for part in parts:
        lowered_part = part.lower()
        if lowered_part == "analyzing":
            continue
        if lowered_part == "working":
            continue
        if lowered_part == "yes":
            normalized_parts.append("Working")
            continue
        if lowered_part == "no":
            normalized_parts.append("Not Working")
            continue
        if lowered_part == "not working":
            normalized_parts.append("Not Working")
            continue
        normalized_parts.append(part.title())

    if "working|yes" in lowered:
        normalized_parts = [part for part in normalized_parts if part != "Not Working"]
    if "working|no" in lowered:
        normalized_parts = [part for part in normalized_parts if part != "Working"]

    ordered_parts = []
    seen = set()
    for part in normalized_parts:
        if part not in seen:
            ordered_parts.append(part)
            seen.add(part)
    return " | ".join(ordered_parts)


def _format_duration_hm(duration_sec: float) -> str:
    total_minutes = int(round(float(duration_sec) / 60))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}h {minutes:02d}m"


def _date_filter_bounds(min_dt: pd.Timestamp, max_dt: pd.Timestamp):
    today = pd.Timestamp.now().date()
    min_date = min_dt.date()
    max_data_date = max_dt.date()
    max_selectable_date = max(today, max_data_date)

    if today < min_date:
        default_date = min_date
    else:
        default_date = today

    return min_date, max_selectable_date, default_date


def _apply_time_period_filter(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df

    min_dt = df["start_datetime"].min()
    max_dt = df["end_datetime"].max()
    min_date, max_selectable_date, default_date = _date_filter_bounds(min_dt, max_dt)

    col1, col2, col3, col4 = st.columns(4)
    start_date = col1.date_input(
        "From Date",
        value=default_date,
        min_value=min_date,
        max_value=max_selectable_date,
        key=f"{key_prefix}_from_date",
    )
    end_date = col2.date_input(
        "To Date",
        value=default_date,
        min_value=min_date,
        max_value=max_selectable_date,
        key=f"{key_prefix}_to_date",
    )
    start_time = col3.time_input(
        "From Time", value=dt_time(0, 0), key=f"{key_prefix}_from_time"
    )
    end_time = col4.time_input(
        "To Time", value=dt_time(23, 59, 59), key=f"{key_prefix}_to_time"
    )

    start_ts = pd.Timestamp(datetime.combine(start_date, start_time))
    end_ts = pd.Timestamp(datetime.combine(end_date, end_time))

    if start_ts > end_ts:
        st.warning("From datetime cannot be after To datetime.")
        return df.iloc[0:0]

    return df[
        (df["start_datetime"] >= start_ts) & (df["start_datetime"] <= end_ts)
    ].copy()


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

        # Expected name: act_<uid>__<name>__<activity_slug>__<epoch_ms>.jpg
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

    # Trim mostly-dark borders so the subject uses more of the preview area.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 16
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        cropped = image[y0:y1, x0:x1]
        if cropped.size:
            image = cropped

    # Fit by height first: top/bottom always filled. Sides may crop or leave gaps.
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


def _normalize_region_slug(value: str) -> str:
    return _slugify_value(value)


def _load_json_file(path: Path):
    if not path.exists() or not path.is_file():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _loads_json_text(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _format_json_list(value, limit: int = 8) -> str:
    data = _loads_json_text(value) if isinstance(value, str) else value
    if not data:
        return ""
    if not isinstance(data, (list, tuple)):
        return str(data)
    items = [str(v).strip() for v in data if str(v).strip()]
    if not items:
        return ""
    shown = items[: int(limit)]
    suffix = f" +{len(items) - len(shown)}" if len(items) > len(shown) else ""
    return ", ".join(shown) + suffix


def _format_json_counts(value, limit: int = 6) -> str:
    data = _loads_json_text(value) if isinstance(value, str) else value
    if not data:
        return ""
    if isinstance(data, list):
        # Some writers store pairs like [["standing|working", 4], ...]
        try:
            as_dict = {str(k): float(v) for k, v in data}
        except Exception:
            return _format_json_list(data, limit=limit)
        data = as_dict
    if not isinstance(data, dict):
        return str(data)
    items = []
    for k, v in data.items():
        key = str(k).strip()
        if not key:
            continue
        try:
            num = float(v)
        except Exception:
            num = 0.0
        items.append((key, num))
    if not items:
        return ""
    items.sort(key=lambda kv: kv[1], reverse=True)
    shown = items[: int(limit)]
    parts = []
    for key, num in shown:
        if abs(num - round(num)) < 1e-9:
            parts.append(f"{key} ({int(round(num))})")
        else:
            parts.append(f"{key} ({num:.2f})")
    suffix = f" +{len(items) - len(shown)}" if len(items) > len(shown) else ""
    return ", ".join(parts) + suffix


def _format_live_region_people(people, limit: int = 6) -> str:
    data = _loads_json_text(people) if isinstance(people, str) else people
    if not data:
        return "None"
    if not isinstance(data, (list, tuple)):
        return str(data)

    items = []
    for person in data:
        if isinstance(person, dict):
            name = str(person.get("name") or person.get("uid") or "Unknown").strip()
            activity = str(person.get("activity") or "").strip()
            pose = str(person.get("pose") or "").strip()
            summary = name
            details = " | ".join([value for value in [activity, pose] if value])
            if details:
                summary = f"{summary} ({details})"
            items.append(summary)
            continue

        text = str(person).strip()
        if text:
            items.append(text)

    if not items:
        return "None"

    shown = items[: int(limit)]
    suffix = f" +{len(items) - len(shown)} more" if len(items) > len(shown) else ""
    return ", ".join(shown) + suffix


def _format_live_region_people_html(people, limit: int = 6) -> str:
    data = _loads_json_text(people) if isinstance(people, str) else people
    if not data:
        return '<div class="region-live-people-empty">No people detected</div>'
    if not isinstance(data, (list, tuple)):
        return f'<div class="region-live-people-empty">{html.escape(str(data))}</div>'

    items = []
    for person in data:
        if isinstance(person, dict):
            name = str(person.get("name") or person.get("uid") or "Unknown").strip()
            activity = _format_person_status_activity(person.get("activity") or "")
            pose = str(person.get("pose") or "").strip().title()
            details = " | ".join([value for value in [activity, pose] if value])
            summary = f"{name} ({details})" if details else name
            items.append(summary)
            continue

        text = str(person).strip()
        if text:
            items.append(text)

    if not items:
        return '<div class="region-live-people-empty">No people detected</div>'

    shown = items[: int(limit)]
    list_items = "".join(
        f'<li>{html.escape(str(item))}</li>' for item in shown
    )
    more_html = (
        f'<div class="region-live-people-more">+{len(items) - len(shown)} more</div>'
        if len(items) > len(shown)
        else ""
    )
    return f'<ul class="region-live-people-list">{list_items}</ul>{more_html}'


def _read_image_rgb(path: Path, retries=3, retry_delay=0.05):
    for _ in range(retries):
        try:
            image_bytes = path.read_bytes()
        except FileNotFoundError:
            time.sleep(retry_delay)
            continue
        except OSError:
            return None

        if not image_bytes:
            time.sleep(retry_delay)
            continue

        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            time.sleep(retry_delay)
            continue

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return None


def _collect_region_live_outputs():
    if not REGION_OUTPUT_DIR.exists():
        return {}

    live_outputs = {}
    for region_dir in REGION_OUTPUT_DIR.iterdir():
        if not region_dir.is_dir():
            continue

        region_slug = region_dir.name
        state = _load_json_file(region_dir / "region_state.json")
        image_candidates = sorted(
            [
                path
                for path in region_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        preferred_name = f"{region_slug}_latest.jpg"
        preferred_image = region_dir / preferred_name
        image_path = (
            preferred_image
            if preferred_image.exists()
            else (image_candidates[0] if image_candidates else None)
        )

        display_name = str(state.get("region_name") or region_slug).strip()
        live_outputs[region_slug] = {
            "region_slug": region_slug,
            "region_name": display_name,
            "state": state,
            "image_path": image_path,
        }

    return live_outputs


def _render_person_activity_table(activity_view: pd.DataFrame):
    rows_html = []
    for _, row in activity_view.iterrows():
        start_time = pd.Timestamp(row["Start Time"]).strftime("%Y-%m-%d %H:%M:%S")
        end_time = pd.Timestamp(row["End Time"]).strftime("%Y-%m-%d %H:%M:%S")
        image_src = str(row["Screenshot"]).strip()
        image_cell = (
            f'<img src="{image_src}" alt="screenshot" class="shot-img" />'
            if image_src
            else '<div class="shot-empty">No image</div>'
        )

        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['Activity']))}</td>"
            f"<td>{html.escape(str(row['Region']))}</td>"
            f"<td>{html.escape(start_time)}</td>"
            f"<td>{html.escape(end_time)}</td>"
            f"<td>{float(row['Duration (sec)']):.2f}</td>"
            f"<td>{float(row['Duration (min)']):.2f}</td>"
            f"<td>{image_cell}</td>"
            "</tr>"
        )

    table_html = f"""
    <style>
      .person-table-wrap {{
        max-height: 760px;
        overflow-y: auto;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
      }}
      .person-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: "Trebuchet MS", "Verdana", sans-serif;
        font-size: 18px;
      }}
      .person-table thead th {{
        position: sticky;
        top: 0;
        z-index: 2;
        background: #111827;
        color: #e5e7eb;
        text-align: left;
        font-size: 18px;
        font-weight: 700;
        padding: 14px 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.35);
      }}
      .person-table td {{
        color: #d1d5db;
        padding: 14px 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        vertical-align: middle;
      }}
      .person-table td:last-child {{
        padding: 8px;
      }}
      .person-table tbody tr:hover {{
        background: rgba(30, 41, 59, 0.45);
      }}
      .shot-img {{
        width: 300px;
        height: 300px;
        object-fit: contain;
        background: rgba(2, 6, 23, 0.55);
        display: block;
        margin: 0 auto;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.4);
      }}
      .shot-empty {{
        width: 300px;
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        border: 1px dashed rgba(148, 163, 184, 0.35);
        color: #9ca3af;
        font-size: 16px;
      }}
    </style>
    <div class="person-table-wrap">
      <table class="person-table">
        <thead>
          <tr>
            <th>Activity</th>
            <th>Region</th>
            <th>Start Time</th>
            <th>End Time</th>
            <th>Duration (sec)</th>
            <th>Duration (min)</th>
            <th>Screenshot</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def _render_region_live_horizontal_table(region_rows):
    if not region_rows:
        st.info("No live region records available.")
        return

    def _image_cell(image_src: str, full_image_src: str, region_name: str):
        image_src = str(image_src).strip()
        if not image_src:
            return '<div class="region-shot-empty">No image</div>'
        region_escaped = html.escape(str(region_name))
        full_image_src = str(full_image_src or image_src).strip() or image_src
        return (
            f'<button type="button" class="region-shot-button" '
            f'onclick="openRegionImage(\'{region_escaped}\', \'{full_image_src}\')">'
            f'<img src="{image_src}" alt="{region_escaped}" class="region-shot-img" />'
            f'<span class="region-shot-overlay">Click for full view</span>'
            f"</button>"
        )

    headers = "".join(
        f"<th>{html.escape(str(row.get('region', '')))}</th>" for row in region_rows
    )
    field_rows = [
        (
            "Live Feed",
            lambda row: _image_cell(
                row.get("image_src", ""),
                row.get("full_image_src", ""),
                row.get("region", ""),
            ),
        ),
        ("Updated At", lambda row: html.escape(str(row.get("updated_at", "")))),
        ("Occupancy", lambda row: str(int(row.get("occupancy", 0) or 0))),
        ("Active", lambda row: html.escape(str(row.get("active", "No")))),
        (
            "People",
            lambda row: (
                f'<div class="region-live-people-cell">'
                f"{_format_live_region_people_html(row.get('people'))}"
                f"</div>"
            ),
        ),
    ]

    body_rows = []
    for label, renderer in field_rows:
        values_html = "".join(f"<td>{renderer(row)}</td>" for row in region_rows)
        body_rows.append(f"<tr><th>{html.escape(label)}</th>{values_html}</tr>")

    table_html = f"""
<style>
  body {{
    margin: 0;
    background: transparent;
  }}
  .region-horizontal-wrap {{
    overflow-x: auto;
    overflow-y: auto;
    max-height: 1100px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 12px;
  }}
  .region-horizontal-table {{
    width: max-content;
    min-width: 100%;
    border-collapse: collapse;
    font-family: "Trebuchet MS", "Verdana", sans-serif;
    font-size: 16px;
    table-layout: fixed;
  }}
  .region-horizontal-table thead th {{
    position: sticky;
    top: 0;
    z-index: 2;
    background: #111827;
    color: #e5e7eb;
    text-align: left;
    font-weight: 700;
    padding: 12px 10px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.35);
    min-width: 320px;
  }}
  .region-horizontal-table tbody th {{
    position: sticky;
    left: 0;
    z-index: 1;
    background: #0f172a;
    color: #e5e7eb;
    text-align: left;
    font-weight: 700;
    padding: 12px 10px;
    border-right: 1px solid rgba(148, 163, 184, 0.25);
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    min-width: 150px;
  }}
  .region-horizontal-table td {{
    color: #d1d5db;
    padding: 12px 10px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    vertical-align: top;
    min-width: 320px;
    white-space: normal;
    word-break: break-word;
  }}
  .region-horizontal-table tbody tr:hover td,
  .region-horizontal-table tbody tr:hover th {{
    background: rgba(30, 41, 59, 0.45);
  }}
  .region-shot-img {{
    width: 300px;
    height: 210px;
    object-fit: cover;
    background: rgba(2, 6, 23, 0.55);
    display: block;
    border-radius: 8px;
    border: 1px solid rgba(148, 163, 184, 0.4);
  }}
  .region-shot-button {{
    position: relative;
    display: inline-flex;
    padding: 0;
    border: none;
    background: transparent;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
  }}
  .region-shot-overlay {{
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(2, 6, 23, 0.68);
    color: #e5e7eb;
    font-size: 15px;
    font-weight: 700;
    opacity: 0;
    transition: opacity 0.18s ease;
  }}
  .region-shot-button:hover .region-shot-overlay {{
    opacity: 1;
  }}
  .region-shot-empty {{
    width: 300px;
    height: 210px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    border: 1px dashed rgba(148, 163, 184, 0.35);
    color: #9ca3af;
    font-size: 14px;
  }}
  .region-live-people-cell {{
    max-width: 320px;
    line-height: 1.45;
    white-space: normal;
    word-break: break-word;
  }}
  .region-live-people-list {{
    margin: 0;
    padding-left: 18px;
  }}
  .region-live-people-list li {{
    margin: 0 0 6px 0;
  }}
  .region-live-people-empty,
  .region-live-people-more {{
    color: #d1d5db;
  }}
  .region-live-people-more {{
    margin-top: 6px;
    font-size: 14px;
    color: #94a3b8;
  }}
  .region-image-modal {{
    position: fixed;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    background: rgba(2, 6, 23, 0.94);
    z-index: 9999;
    padding: 8px;
    box-sizing: border-box;
  }}
  .region-image-modal.is-open {{
    display: flex;
  }}
  .region-image-modal__content {{
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }}
  .region-image-modal__topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 4px 6px 0;
  }}
  .region-image-modal__title {{
    color: #e5e7eb;
    font-family: "Trebuchet MS", "Verdana", sans-serif;
    font-size: 32px;
    font-weight: 700;
  }}
  .region-image-modal__close {{
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: rgba(15, 23, 42, 0.88);
    color: #e5e7eb;
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
  }}
  .region-image-modal__viewport {{
    flex: 1;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 14px;
    background: rgba(15, 23, 42, 0.45);
    overflow: auto;
    padding: 8px;
  }}
  .region-image-modal__img {{
    width: auto;
    height: auto;
    max-width: min(98vw, 1800px);
    max-height: calc(100vh - 110px);
    object-fit: contain;
    display: block;
  }}
</style>
<div class="region-horizontal-wrap">
  <table class="region-horizontal-table">
    <thead>
      <tr>
        <th>Field</th>
        {headers}
      </tr>
    </thead>
    <tbody>
      {"".join(body_rows)}
    </tbody>
  </table>
</div>
<div id="region-image-modal" class="region-image-modal" onclick="closeRegionImage(event)">
  <div class="region-image-modal__content">
    <div class="region-image-modal__topbar">
      <div id="region-image-modal-title" class="region-image-modal__title">Region Image</div>
      <button type="button" class="region-image-modal__close" onclick="closeRegionImage(event)">Back</button>
    </div>
    <div class="region-image-modal__viewport">
      <img id="region-image-modal-img" class="region-image-modal__img" alt="Region full view" />
    </div>
  </div>
</div>
<script>
  function openRegionImage(regionName, imageSrc) {{
    var modal = document.getElementById("region-image-modal");
    var title = document.getElementById("region-image-modal-title");
    var image = document.getElementById("region-image-modal-img");
    if (!modal || !title || !image) return;
    title.textContent = regionName;
    image.src = imageSrc;
    image.alt = regionName + " full view";
    modal.classList.add("is-open");
    document.body.style.overflow = "hidden";
  }}

  function closeRegionImage(event) {{
    if (event) {{
      event.preventDefault();
      event.stopPropagation();
    }}
    var modal = document.getElementById("region-image-modal");
    var image = document.getElementById("region-image-modal-img");
    if (!modal || !image) return;
    modal.classList.remove("is-open");
    image.src = "";
    document.body.style.overflow = "";
  }}

  document.addEventListener("keydown", function(event) {{
    if (event.key === "Escape") {{
      closeRegionImage();
    }}
  }});
</script>
"""
    components.html(table_html, height=1500, scrolling=False)


@st.cache_resource(show_spinner=False)
def _load_haar_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    return detector


def _extract_largest_face_jpg(image_bytes: bytes):
    if not image_bytes:
        return None, None, "No image provided.", None

    img_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, "Invalid image format.", None

    detector = _load_haar_face_detector()
    if detector.empty():
        return None, None, "Face detector is not available.", None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return frame, None, "No face detected.", None

    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return frame, None, "Face crop failed.", (x, y, w, h)

    ok, encoded = cv2.imencode(".jpg", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return frame, None, "Face encoding failed.", (x, y, w, h)

    return frame, encoded.tobytes(), "", (x, y, w, h)


def _build_face_guided_preview(frame: np.ndarray, bbox):
    settings = _face_registration_settings()
    preview = frame.copy()
    h, w = preview.shape[:2]

    guide_w = int(w * float(settings["guide_width_ratio"]))
    guide_h = int(h * float(settings["guide_height_ratio"]))
    gx1 = (w - guide_w) // 2
    gy1 = (h - guide_h) // 2
    gx2 = gx1 + guide_w
    gy2 = gy1 + guide_h

    cv2.rectangle(preview, (gx1, gy1), (gx2, gy2), (255, 180, 0), 3)

    if bbox is None:
        return preview, "No face detected. Put your face inside the box.", False

    x, y, bw, bh = [int(v) for v in bbox]
    fx1, fy1, fx2, fy2 = x, y, x + bw, y + bh
    face_cx = (fx1 + fx2) // 2
    face_cy = (fy1 + fy2) // 2
    in_guide = gx1 <= face_cx <= gx2 and gy1 <= face_cy <= gy2
    face_area = max(1, bw * bh)
    guide_area = max(1, guide_w * guide_h)
    size_ratio = face_area / float(guide_area)
    good_size = (
        float(settings["guide_min_size_ratio"])
        <= size_ratio
        <= float(settings["guide_max_size_ratio"])
    )
    is_good = in_guide and good_size

    box_color = (0, 220, 90) if is_good else (0, 120, 255)
    cv2.rectangle(preview, (fx1, fy1), (fx2, fy2), box_color, 3)

    if not in_guide:
        msg = "Move face to the center guide box."
    elif size_ratio < float(settings["guide_min_size_ratio"]):
        msg = "Move closer to camera."
    elif size_ratio > float(settings["guide_max_size_ratio"]):
        msg = "Move slightly back from camera."
    else:
        msg = "Great alignment. Capture this sample."
    return preview, msg, is_good


def _create_mediapipe_face_mesh():
    if mp is None:
        return None

    # Common API (mediapipe solutions package)
    solutions = getattr(mp, "solutions", None)
    face_mesh_mod = (
        getattr(solutions, "face_mesh", None) if solutions is not None else None
    )
    if face_mesh_mod is not None and hasattr(face_mesh_mod, "FaceMesh"):
        return face_mesh_mod.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # Compatibility fallback for some package layouts.
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh  # type: ignore

        return mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_mediapipe_face_mesh():
    return _create_mediapipe_face_mesh()


def _estimate_head_pose_from_face_mesh(landmarks, img_w: int, img_h: int):
    settings = _face_registration_settings()
    # Landmark ids from MediaPipe Face Mesh.
    nose = landmarks[1]
    left_eye_outer = landmarks[33]
    right_eye_outer = landmarks[263]
    chin = landmarks[152]
    forehead = landmarks[10]

    nx, ny = nose.x * img_w, nose.y * img_h
    lex, ley = left_eye_outer.x * img_w, left_eye_outer.y * img_h
    rex, rey = right_eye_outer.x * img_w, right_eye_outer.y * img_h
    _, chy = chin.x * img_w, chin.y * img_h
    _, fhy = forehead.x * img_w, forehead.y * img_h

    eye_mid_x = (lex + rex) / 2.0
    eye_mid_y = (ley + rey) / 2.0
    eye_dist = max(1.0, abs(rex - lex))
    face_h = max(1.0, abs(chy - fhy))

    yaw = (nx - eye_mid_x) / eye_dist
    pitch = (ny - eye_mid_y) / face_h

    if yaw <= float(settings["fallback_left_yaw_ratio"]):
        pose = "Left Profile"
    elif yaw >= float(settings["fallback_right_yaw_ratio"]):
        pose = "Right Profile"
    elif pitch >= float(settings["fallback_up_pitch_ratio"]):
        pose = "Up Tilt"
    elif pitch <= float(settings["fallback_down_pitch_ratio"]):
        pose = "Down Tilt"
    else:
        pose = "Frontal"
    return pose, float(yaw), float(pitch)


def _extract_face_crop_from_mesh(frame: np.ndarray, landmarks, expand: float | None = None):
    settings = _face_registration_settings()
    if expand is None:
        expand = float(settings["face_crop_expand_ratio"])
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = int(max(0, min(xs))), int(min(w, max(xs)))
    y1, y2 = int(max(0, min(ys))), int(min(h, max(ys)))

    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    padx, pady = int(bw * expand), int(bh * expand)
    x1 = max(0, x1 - padx)
    y1 = max(0, y1 - pady)
    x2 = min(w, x2 + padx)
    y2 = min(h, y2 + pady)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None
    ok, encoded = cv2.imencode(".jpg", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return None, None
    return encoded.tobytes(), (x1, y1, x2, y2)


def _extract_face_with_mediapipe_or_haar(image_bytes: bytes):
    if not image_bytes:
        return None, None, "No image provided.", None, None, "None"

    img_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, "Invalid image format.", None, None, "None"

    # Try MediaPipe first for face mesh + pose estimation.
    mesh = _load_mediapipe_face_mesh()
    if mesh is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)
        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            detected_pose, _, _ = _estimate_head_pose_from_face_mesh(
                lms, frame.shape[1], frame.shape[0]
            )
            face_bytes, face_box = _extract_face_crop_from_mesh(frame, lms)
            if face_bytes is not None:
                return frame, face_bytes, "", face_box, detected_pose, "MediaPipe"

    # Fallback to Haar detector to keep functionality if MediaPipe is unavailable.
    frame2, face_bytes2, err2, bbox2 = _extract_largest_face_jpg(image_bytes)
    return frame2, face_bytes2, err2, bbox2, None, "Haar"


def _estimate_head_angles_from_mesh(landmarks, img_w: int, img_h: int):
    eye_mid_y = ((landmarks[33].y + landmarks[263].y) * 0.5) * img_h
    mouth_mid_y = ((landmarks[61].y + landmarks[291].y) * 0.5) * img_h
    nose_y = landmarks[1].y * img_h
    denom = max(1.0, mouth_mid_y - eye_mid_y)
    vertical_ratio = float((nose_y - eye_mid_y) / denom)

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (-35.0, -30.0, -30.0),
            (35.0, -30.0, -30.0),
            (-70.0, -30.0, -60.0),
            (70.0, -30.0, -60.0),
        ],
        dtype=np.float64,
    )
    image_points = np.array(
        [
            [landmarks[1].x * img_w, landmarks[1].y * img_h],  # nose
            [landmarks[33].x * img_w, landmarks[33].y * img_h],  # left eye outer
            [landmarks[263].x * img_w, landmarks[263].y * img_h],  # right eye outer
            [landmarks[61].x * img_w, landmarks[61].y * img_h],  # mouth left
            [landmarks[291].x * img_w, landmarks[291].y * img_h],  # mouth right
        ],
        dtype=np.float64,
    )

    camera_matrix = np.array(
        [
            [img_w, 0, img_w / 2],
            [0, img_w, img_h / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None, None, vertical_ratio

    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return float(yaw), float(pitch), vertical_ratio


def _classify_direction_from_angles(
    yaw: float,
    pitch: float,
    last_vertical: str,
    vertical_ratio: float | None = None,
    baseline_vertical_ratio: float | None = None,
    baseline_yaw: float | None = None,
    baseline_pitch: float | None = None,
):
    settings = _face_registration_settings()
    frontal_yaw_deadzone = float(settings["frontal_yaw_deadzone"])
    frontal_pitch_deadzone = float(settings["frontal_pitch_deadzone"])
    yaw_threshold = float(settings["raw_yaw_threshold"])
    pitch_up_threshold = float(settings["raw_pitch_up_threshold"])
    pitch_down_threshold = float(settings["raw_pitch_down_threshold"])
    pitch_deadzone = float(settings["raw_pitch_deadzone"])

    yaw_value = float(yaw)
    if baseline_yaw is not None:
        yaw_value = float(yaw) - float(baseline_yaw)

    pitch_value = float(pitch)
    if baseline_pitch is not None:
        pitch_value = float(pitch) - float(baseline_pitch)

    if baseline_yaw is not None:
        if yaw_value <= float(settings["baseline_left_threshold"]):
            horizontal = "Left"
        elif yaw_value >= float(settings["baseline_right_threshold"]):
            horizontal = "Right"
        else:
            horizontal = "Forward"
    else:
        if yaw_value > yaw_threshold:
            horizontal = "Left"
        elif yaw_value < -yaw_threshold:
            horizontal = "Right"
        else:
            horizontal = "Forward"

    if (
        baseline_vertical_ratio is not None
        and vertical_ratio is not None
        and baseline_pitch is not None
    ):
        delta = vertical_ratio - baseline_vertical_ratio
        if (
            pitch_value >= float(settings["baseline_up_pitch_threshold"])
            or delta <= float(settings["baseline_up_vertical_ratio_delta"])
        ):
            vertical = "Up"
        elif (
            pitch_value <= float(settings["baseline_down_pitch_threshold"])
            or delta >= float(settings["baseline_down_vertical_ratio_delta"])
        ):
            vertical = "Down"
        elif abs(pitch_value) <= frontal_pitch_deadzone and abs(delta) <= float(
            settings["baseline_level_vertical_ratio_deadzone"]
        ):
            vertical = "Level"
        else:
            vertical = last_vertical
    elif baseline_vertical_ratio is not None and vertical_ratio is not None:
        delta = vertical_ratio - baseline_vertical_ratio
        if delta <= float(settings["baseline_up_vertical_ratio_delta"]):
            vertical = "Up"
        elif delta >= float(settings["baseline_down_vertical_ratio_delta"]):
            vertical = "Down"
        elif abs(delta) <= float(settings["baseline_level_vertical_ratio_deadzone"]):
            vertical = "Level"
        else:
            vertical = last_vertical
    else:
        # Before the user's frontal sample is captured, there is no reliable
        # per-person baseline for vertical pose. Keep the initial capture
        # stable by only classifying the horizontal axis at this stage.
        vertical = "Level"

    if abs(yaw_value) <= frontal_yaw_deadzone and vertical == "Level":
        return "Forward-Level"

    return f"{horizontal}-{vertical}"


def _pose_matches_target(
    target_pose: str, current_pose: str, current_direction: str
) -> bool:
    horizontal, _, vertical = str(current_direction).partition("-")
    if target_pose == "Frontal":
        return horizontal == "Forward" and vertical == "Level"
    if target_pose == "Left Profile":
        return horizontal == "Left"
    if target_pose == "Right Profile":
        return horizontal == "Right"
    if target_pose == "Up Tilt":
        return horizontal == "Forward" and vertical == "Up"
    if target_pose == "Down Tilt":
        return horizontal == "Forward" and vertical == "Down"
    return current_pose == target_pose


def _direction_to_pose(direction: str):
    horizontal = direction.split("-")[0] if "-" in direction else direction
    vertical = direction.split("-")[1] if "-" in direction else "Level"
    if horizontal == "Left":
        return "Left Profile"
    if horizontal == "Right":
        return "Right Profile"
    if vertical == "Up":
        return "Up Tilt"
    if vertical == "Down":
        return "Down Tilt"
    return "Frontal"


class ThreadedFaceCaptureProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame_queue = deque(maxlen=1)
        self._annotated_frame = None
        self._latest_state = {
            "current_pose": "No Face",
            "current_direction": "No Face",
            "yaw": None,
            "pitch": None,
            "face_bytes": None,
            "face_preview": None,
            "vertical_ratio": None,
            "aligned": False,
            "face_centered": False,
            "guide_message": "Waiting for face detection.",
            "error": "",
        }
        self._target_pose = "Frontal"
        self._baseline_vertical_ratio = None
        self._baseline_yaw = None
        self._baseline_pitch = None
        self._vote_buffer = []
        settings = _face_registration_settings()
        self._vote_window = int(settings["pose_vote_window"])
        self._angle_smooth = float(settings["angle_smooth_factor"])
        self._last_yaw = None
        self._last_pitch = None
        self._last_direction = "Forward-Level"
        self._mesh = _create_mediapipe_face_mesh()
        self._worker = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker.start()

    def update_guidance(
        self, target_pose: str, baseline_vertical_ratio, baseline_yaw, baseline_pitch
    ):
        with self._lock:
            self._target_pose = str(target_pose)
            self._baseline_vertical_ratio = baseline_vertical_ratio
            self._baseline_yaw = baseline_yaw
            self._baseline_pitch = baseline_pitch

    def get_state(self):
        with self._lock:
            state = dict(self._latest_state)
            if isinstance(self._annotated_frame, np.ndarray):
                state["annotated_frame"] = self._annotated_frame.copy()
            else:
                state["annotated_frame"] = None
            return state

    def _processing_loop(self):
        while not self._stop_event.is_set():
            frame = None
            with self._lock:
                if self._frame_queue:
                    frame = self._frame_queue.pop()
                    self._frame_queue.clear()
                    target_pose = self._target_pose
                    baseline_vertical_ratio = self._baseline_vertical_ratio
                    baseline_yaw = self._baseline_yaw
                    baseline_pitch = self._baseline_pitch
                else:
                    target_pose = self._target_pose
                    baseline_vertical_ratio = self._baseline_vertical_ratio
                    baseline_yaw = self._baseline_yaw
                    baseline_pitch = self._baseline_pitch

            if frame is None:
                time.sleep(0.01)
                continue

            annotated = frame.copy()
            current_pose = "No Face"
            current_direction = "No Face"
            yaw_value = None
            pitch_value = None
            face_bytes = None
            face_preview = None
            vertical_ratio = None
            aligned = False
            face_centered = False
            guide_message = "No face detected. Put your face inside the box."
            error = ""
            bbox = None

            if self._mesh is None:
                error = "MediaPipe Face Mesh is not available."
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self._mesh.process(rgb)
                if result.multi_face_landmarks:
                    settings = _face_registration_settings()
                    self._vote_window = int(settings["pose_vote_window"])
                    self._angle_smooth = float(settings["angle_smooth_factor"])
                    lms = result.multi_face_landmarks[0].landmark
                    yaw, pitch, vertical_ratio = _estimate_head_angles_from_mesh(
                        lms,
                        frame.shape[1],
                        frame.shape[0],
                    )
                    face_bytes, bbox = _extract_face_crop_from_mesh(frame, lms)

                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        face_preview = frame[y1:y2, x1:x2].copy()
                        annotated, guide_message, aligned = _build_face_guided_preview(
                            frame,
                            (x1, y1, x2 - x1, y2 - y1),
                        )
                        face_centered = (
                            "Move face to the center guide box." != guide_message
                        )

                    if yaw is not None and pitch is not None:
                        if self._last_yaw is not None:
                            yaw = (
                                self._angle_smooth * self._last_yaw
                                + (1 - self._angle_smooth) * yaw
                            )
                            pitch = (
                                self._angle_smooth * self._last_pitch
                                + (1 - self._angle_smooth) * pitch
                            )
                        self._last_yaw, self._last_pitch = yaw, pitch
                        yaw_value = float(yaw)
                        pitch_value = float(pitch)

                        last_vertical = (
                            self._last_direction.split("-")[1]
                            if "-" in self._last_direction
                            else "Level"
                        )
                        direction = _classify_direction_from_angles(
                            yaw,
                            pitch,
                            last_vertical,
                            vertical_ratio=vertical_ratio,
                            baseline_vertical_ratio=baseline_vertical_ratio,
                            baseline_yaw=baseline_yaw,
                            baseline_pitch=baseline_pitch,
                        )
                        self._vote_buffer.append(direction)
                        # Frontal is the neutral pose and should lock quickly once
                        # the user is centered. Waiting for a long vote window makes
                        # it feel sluggish compared with stronger side/up/down poses.
                        if target_pose == "Frontal" and direction == "Forward-Level":
                            current_direction = direction
                            self._vote_buffer.clear()
                            self._last_direction = current_direction
                        elif len(self._vote_buffer) >= self._vote_window:
                            current_direction = Counter(self._vote_buffer).most_common(
                                1
                            )[0][0]
                            self._vote_buffer.clear()
                            self._last_direction = current_direction
                        else:
                            current_direction = self._last_direction
                        current_pose = _direction_to_pose(current_direction)
                    else:
                        fallback_pose, fallback_yaw, fallback_pitch = (
                            _estimate_head_pose_from_face_mesh(
                                lms,
                                frame.shape[1],
                                frame.shape[0],
                            )
                        )
                        yaw_value = fallback_yaw
                        pitch_value = fallback_pitch
                        current_pose = fallback_pose
                        if fallback_pose == "Left Profile":
                            current_direction = "Left-Level"
                        elif fallback_pose == "Right Profile":
                            current_direction = "Right-Level"
                        elif fallback_pose == "Up Tilt":
                            current_direction = "Forward-Up"
                        elif fallback_pose == "Down Tilt":
                            current_direction = "Forward-Down"
                        else:
                            current_direction = "Forward-Level"
                        self._last_direction = current_direction
                else:
                    self._vote_buffer.clear()

            text_color = (0, 255, 0) if current_pose == target_pose else (0, 200, 255)
            cv2.putText(
                annotated,
                f"Detected: {current_pose} ({current_direction})",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                text_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Target: {target_pose}",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 220, 0),
                2,
                cv2.LINE_AA,
            )
            if error:
                cv2.putText(
                    annotated,
                    error,
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 120, 255),
                    2,
                    cv2.LINE_AA,
                )

            with self._lock:
                self._annotated_frame = annotated
                self._latest_state = {
                    "current_pose": current_pose,
                    "current_direction": current_direction,
                    "yaw": yaw_value,
                    "pitch": pitch_value,
                    "face_bytes": face_bytes,
                    "face_preview": face_preview,
                    "vertical_ratio": vertical_ratio,
                    "aligned": aligned,
                    "face_centered": face_centered,
                    "guide_message": guide_message,
                    "error": error,
                }

    def recv(self, frame):
        if av is None:
            return frame

        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        with self._lock:
            self._frame_queue.append(image)
            annotated = (
                self._annotated_frame.copy()
                if isinstance(self._annotated_frame, np.ndarray)
                else image
            )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ==============================
# DATABASE LAYER
# ==============================


class SurveillanceRepository:
    def __init__(self, face_db_path, activity_db_path):
        self.face_db = face_db_path
        self.activity_db = activity_db_path

    def _ensure_users_schema(self, conn: sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "username TEXT UNIQUE, "
            "embedding BLOB)"
        )
        cursor.execute("PRAGMA table_info(users)")
        cols = {c[1] for c in cursor.fetchall()}
        if "created_at" not in cols:
            cursor.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
            cursor.execute(
                "UPDATE users SET created_at = datetime('now') WHERE created_at IS NULL"
            )
        if "face_image" not in cols:
            cursor.execute("ALTER TABLE users ADD COLUMN face_image BLOB")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS face_samples ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "username TEXT NOT NULL, "
            "face_image BLOB NOT NULL, "
            "pose TEXT, "
            "lighting TEXT, "
            "glasses INTEGER DEFAULT 0, "
            "created_at TEXT DEFAULT (datetime('now'))"
            ")"
        )
        conn.commit()

    def get_faces(self):
        with sqlite3.connect(self.face_db) as conn:
            self._ensure_users_schema(conn)
            return pd.read_sql_query(
                "SELECT username, created_at, face_image FROM users", conn
            )

    def upsert_face_image(self, username: str, face_image: bytes):
        clean_username = str(username).strip()
        if not clean_username:
            return False, "Username is required."
        if not face_image:
            return False, "Face image is required."

        try:
            with sqlite3.connect(self.face_db) as conn:
                self._ensure_users_schema(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO users (username, face_image, created_at) "
                    "VALUES (?, ?, datetime('now'))",
                    (clean_username, face_image),
                )
                conn.commit()
        except sqlite3.Error as exc:
            return False, f"Database error: {exc}"

        return True, f"Face saved for {clean_username}."

    def add_face_samples(self, username: str, samples):
        clean_username = str(username).strip()
        if not clean_username:
            return False, "Username is required."
        if not samples:
            return False, "At least one sample is required."

        try:
            with sqlite3.connect(self.face_db) as conn:
                self._ensure_users_schema(conn)
                for sample in samples:
                    face_image = sample.get("face_image")
                    if not face_image:
                        continue
                    conn.execute(
                        "INSERT INTO face_samples (username, face_image, pose, lighting, glasses, created_at) "
                        "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                        (
                            clean_username,
                            face_image,
                            str(sample.get("pose", "")),
                            str(sample.get("lighting", "")),
                            int(bool(sample.get("glasses", False))),
                        ),
                    )

                # Keep latest sample mirrored in users.face_image for compatibility.
                latest_face = samples[-1].get("face_image")
                if latest_face:
                    conn.execute(
                        "INSERT OR REPLACE INTO users (username, face_image, created_at) "
                        "VALUES (?, ?, datetime('now'))",
                        (clean_username, latest_face),
                    )
                conn.commit()
        except sqlite3.Error as exc:
            return False, f"Database error: {exc}"

        return True, f"Saved {len(samples)} samples for {clean_username}."

    def get_face_sample_counts(self):
        with sqlite3.connect(self.face_db) as conn:
            self._ensure_users_schema(conn)
            return pd.read_sql_query(
                """
                SELECT
                    u.username AS username,
                    COUNT(s.id) AS sample_count,
                    MAX(s.created_at) AS last_sample_at
                FROM users u
                LEFT JOIN face_samples s ON s.username = u.username
                GROUP BY u.username
                ORDER BY COALESCE(MAX(s.created_at), u.created_at) DESC
                """,
                conn,
            )

    def get_latest_pose_samples(self):
        with sqlite3.connect(self.face_db) as conn:
            self._ensure_users_schema(conn)
            return pd.read_sql_query(
                """
                SELECT s.username, s.pose, s.face_image, s.created_at
                FROM face_samples s
                JOIN (
                    SELECT username, pose, MAX(id) AS max_id
                    FROM face_samples
                    GROUP BY username, pose
                ) latest
                ON s.id = latest.max_id
                ORDER BY s.username, s.pose
                """,
                conn,
            )

    def delete_face(self, username: str):
        clean_username = str(username).strip()
        if not clean_username:
            return False, "Username is required."
        try:
            with sqlite3.connect(self.face_db) as conn:
                self._ensure_users_schema(conn)
                conn.execute(
                    "DELETE FROM face_samples WHERE username = ?", (clean_username,)
                )
                conn.execute("DELETE FROM users WHERE username = ?", (clean_username,))
                conn.commit()
        except sqlite3.Error as exc:
            return False, f"Database error: {exc}"
        return True, f"Deleted {clean_username}."

    def rename_face(self, current_username: str, new_username: str):
        clean_current = str(current_username).strip()
        clean_new = str(new_username).strip()
        if not clean_current:
            return False, "Current username is required."
        if not clean_new:
            return False, "New username is required."
        if clean_current == clean_new:
            return False, "Enter a different name."

        try:
            with sqlite3.connect(self.face_db) as conn:
                self._ensure_users_schema(conn)
                existing = conn.execute(
                    "SELECT 1 FROM users WHERE username = ?",
                    (clean_current,),
                ).fetchone()
                if not existing:
                    return False, f"{clean_current} was not found."

                duplicate = conn.execute(
                    "SELECT 1 FROM users WHERE username = ?",
                    (clean_new,),
                ).fetchone()
                if duplicate:
                    return False, f"{clean_new} already exists."

                conn.execute(
                    "UPDATE users SET username = ? WHERE username = ?",
                    (clean_new, clean_current),
                )
                conn.execute(
                    "UPDATE face_samples SET username = ? WHERE username = ?",
                    (clean_new, clean_current),
                )
                conn.commit()
        except sqlite3.Error as exc:
            return False, f"Database error: {exc}"

        return True, f"Renamed {clean_current} to {clean_new}."

    def get_logs(self, page: int = 1, per_page: int = 100, filters: dict = None):
        offset = (page - 1) * per_page

        where_clauses = []
        params = []

        if filters:
            if filters.get("username"):
                where_clauses.append("username = ?")
                params.append(filters["username"])
            if filters.get("start_date"):
                where_clauses.append("start_datetime >= ?")
                params.append(filters["start_date"])
            if filters.get("end_date"):
                where_clauses.append("start_datetime <= ?")
                params.append(filters["end_date"])
            if filters.get("activity"):
                where_clauses.append("activity = ?")
                params.append(filters["activity"])

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        query = f"SELECT * FROM activity_logs{where_sql} ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([per_page, offset])

        with sqlite3.connect(self.activity_db) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["start_datetime"] = pd.to_datetime(df["start_datetime"])
            df["end_datetime"] = pd.to_datetime(df["end_datetime"])
        return df

    def get_logs_count(self, filters: dict = None):
        where_clauses = []
        params = []

        if filters:
            if filters.get("username"):
                where_clauses.append("username = ?")
                params.append(filters["username"])
            if filters.get("start_date"):
                where_clauses.append("start_datetime >= ?")
                params.append(filters["start_date"])
            if filters.get("end_date"):
                where_clauses.append("start_datetime <= ?")
                params.append(filters["end_date"])
            if filters.get("activity"):
                where_clauses.append("activity = ?")
                params.append(filters["activity"])

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        query = f"SELECT COUNT(*) FROM activity_logs{where_sql}"

        with sqlite3.connect(self.activity_db) as conn:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()
            return result[0] if result else 0

    def get_region_chunk_logs(self, page: int = 1, per_page: int = 100, filters: dict = None):
        offset = (page - 1) * per_page

        where_clauses = []
        params = []

        if filters:
            if filters.get("region"):
                where_clauses.append("region = ?")
                params.append(filters["region"])
            if filters.get("start_date"):
                where_clauses.append("chunk_start_datetime >= ?")
                params.append(filters["start_date"])
            if filters.get("end_date"):
                where_clauses.append("chunk_start_datetime <= ?")
                params.append(filters["end_date"])

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        query = f"SELECT * FROM region_chunk_logs{where_sql} ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([per_page, offset])

        try:
            with sqlite3.connect(self.activity_db) as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except (sqlite3.Error, pd.errors.DatabaseError):
            return pd.DataFrame()

        if df.empty:
            return df

        for col in ("chunk_start_datetime", "chunk_end_datetime", "created_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def get_region_chunk_logs_count(self, filters: dict = None):
        where_clauses = []
        params = []

        if filters:
            if filters.get("region"):
                where_clauses.append("region = ?")
                params.append(filters["region"])
            if filters.get("start_date"):
                where_clauses.append("chunk_start_datetime >= ?")
                params.append(filters["start_date"])
            if filters.get("end_date"):
                where_clauses.append("chunk_start_datetime <= ?")
                params.append(filters["end_date"])

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        query = f"SELECT COUNT(*) FROM region_chunk_logs{where_sql}"

        try:
            with sqlite3.connect(self.activity_db) as conn:
                cursor = conn.execute(query, params)
                result = cursor.fetchone()
                return result[0] if result else 0
        except sqlite3.Error:
            return 0

    def delete_activity_records(self, record_ids):
        if record_ids is None:
            return False, "No records selected."
        try:
            ids = [int(rid) for rid in record_ids if pd.notna(rid)]
        except (TypeError, ValueError):
            return False, "Invalid record IDs."
        if not ids:
            return False, "No records selected."

        placeholders = ",".join(["?"] * len(ids))
        try:
            with sqlite3.connect(self.activity_db) as conn:
                cur = conn.execute(
                    f"DELETE FROM activity_logs WHERE id IN ({placeholders})",
                    ids,
                )
                conn.commit()
                deleted = int(cur.rowcount or 0)
        except sqlite3.Error as exc:
            return False, f"Database error: {exc}"

        if deleted == 0:
            return False, "No matching records were deleted."
        return True, f"Deleted {deleted} activity record(s)."


# ==============================
# SERVICE LAYER
# ==============================


class SurveillanceService:
    def __init__(self, repo: SurveillanceRepository):
        self.repo = repo

    def load_data(self, page: int = 1, per_page: int = 100, filters: dict = None):
        return self.repo.get_faces(), self.repo.get_logs(page=page, per_page=per_page, filters=filters)

    def get_faces(self):
        return self.repo.get_faces()

    def save_face(self, username: str, face_image: bytes):
        return self.repo.upsert_face_image(username, face_image)

    def save_face_samples(self, username: str, samples):
        return self.repo.add_face_samples(username, samples)

    def delete_face(self, username: str):
        return self.repo.delete_face(username)

    def get_face_sample_counts(self):
        return self.repo.get_face_sample_counts()

    def get_latest_pose_samples(self):
        return self.repo.get_latest_pose_samples()

    def rename_face(self, current_username: str, new_username: str):
        return self.repo.rename_face(current_username, new_username)

    def delete_activity_records(self, record_ids):
        return self.repo.delete_activity_records(record_ids)

    def get_logs(self, page: int = 1, per_page: int = 100, filters: dict = None):
        return self.repo.get_logs(page=page, per_page=per_page, filters=filters)

    def get_logs_count(self, filters: dict = None):
        return self.repo.get_logs_count(filters=filters)

    def get_region_chunk_logs(self, page: int = 1, per_page: int = 100, filters: dict = None):
        return self.repo.get_region_chunk_logs(page=page, per_page=per_page, filters=filters)

    def get_region_chunk_logs_count(self, filters: dict = None):
        return self.repo.get_region_chunk_logs_count(filters=filters)


# ==============================
# BASE PAGE
# ==============================


class BasePage(ABC):
    def __init__(self, title, service):
        self.title = title
        self.service = service

    def show_title(self):
        st.subheader(self.title)

    @abstractmethod
    def render(self):
        pass


class BaseSection(ABC):
    def __init__(self, service):
        self.service = service

    @abstractmethod
    def render(self):
        pass


# ==============================
# PAGES
# ==============================


class OverviewPage(BasePage):
    def __init__(self, service):
        super().__init__("📊 System Overview", service)

    def render(self):
        self.show_title()
        faces_df, logs_df = self.service.load_data()
        filtered_logs_df = _exclude_analyzing(logs_df)
        filtered_logs_df = _apply_time_period_filter(filtered_logs_df, "overview")

        if filtered_logs_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Registered Users", len(faces_df))
            col2.metric("Unique Persons", 0)
            col3.metric("Total Hours", "0.0")
            col4.metric("Top Activity", "N/A")
            st.info("No overview data is available for the selected date and time range.")
            return

        overview_df = filtered_logs_df.copy()
        overview_df["activity"] = overview_df["activity"].apply(_canonical_activity_label)
        overview_df = overview_df[overview_df["activity"].notna()].copy()

        if overview_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Registered Users", len(faces_df))
            col2.metric("Unique Persons", 0)
            col3.metric("Total Hours", "0.0")
            col4.metric("Active Regions", 0)
            st.info(
                "No overview data matches the selected categories: Working, Not Working, Using Mobile, Sleeping, Head Down."
            )
            return

        overview_df["region_name"] = (
            overview_df["region_name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
        if "name" in overview_df.columns:
            overview_df["person_label"] = (
                overview_df["name"].fillna("").astype(str).str.strip().replace("", pd.NA)
            )
        else:
            overview_df["person_label"] = pd.NA
        overview_df["person_label"] = overview_df["person_label"].fillna(
            overview_df["uid"].astype(str)
        )
        overview_df["hour_bucket"] = overview_df["start_datetime"].dt.floor("h")
        overview_df["hour_of_day"] = overview_df["start_datetime"].dt.strftime("%H:00")
        overview_df["duration_hours"] = overview_df["duration_sec"].clip(lower=0) / 3600.0

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Registered Users", len(faces_df))
        col2.metric("Unique Persons", overview_df["uid"].nunique())
        col3.metric(
            "Total Hours", f"{overview_df['duration_sec'].sum() / 3600:.1f}"
        )
        col4.metric("Active Regions", overview_df["region_name"].nunique())

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Total Records", len(overview_df))
        top_activity = (
            overview_df["activity"].value_counts().idxmax() if not overview_df.empty else "N/A"
        )
        col6.metric("Top Activity", top_activity)
        top_region = (
            overview_df["region_name"].value_counts().idxmax()
            if not overview_df.empty
            else "N/A"
        )
        col7.metric("Top Region", top_region)
        col8.metric(
            "Avg Session",
            _format_duration_hm(overview_df["duration_sec"].mean()) if not overview_df.empty else "00h 00m",
        )

        st.divider()

        trend_df = (
            overview_df.groupby("hour_bucket", as_index=False)
            .agg(active_persons=("uid", "nunique"), records=("uid", "size"))
            .sort_values("hour_bucket")
        )
        colA, colB = st.columns([1.7, 1.3])
        with colA:
            trend_fig = px.line(
                trend_df,
                x="hour_bucket",
                y="active_persons",
                markers=True,
                title="Active Persons Trend",
            )
            trend_fig.update_traces(
                line=dict(width=3, color="#38bdf8"),
                marker=dict(size=8, color="#f59e0b"),
                hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Active Persons: %{y}<extra></extra>",
            )
            trend_fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Active Persons",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(trend_fig, use_container_width=True)

        with colB:
            act_dist = (
                overview_df.groupby("activity", as_index=False)["duration_sec"]
                .sum()
                .sort_values("duration_sec", ascending=False)
            )
            act_dist["duration_label"] = act_dist["duration_sec"].apply(_format_duration_hm)
            activity_fig = px.pie(
                act_dist,
                names="activity",
                values="duration_sec",
                hole=0.55,
                title="Activity Distribution",
            )
            activity_fig.update_traces(
                textinfo="percent+label",
                hovertemplate="Activity: %{label}<br>Total Time: %{value:.0f} sec<extra></extra>",
            )
            activity_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(activity_fig, use_container_width=True)

        region_dist = (
            overview_df.groupby("region_name", as_index=False)["duration_sec"]
            .sum()
            .sort_values("duration_sec", ascending=False)
        )
        region_dist["duration_hours"] = region_dist["duration_sec"] / 3600.0
        region_dist["duration_label"] = region_dist["duration_sec"].apply(_format_duration_hm)

        heatmap_df = (
            overview_df.groupby(["region_name", "hour_of_day"], as_index=False)["uid"]
            .nunique()
            .rename(columns={"uid": "active_persons"})
        )

        colC, colD = st.columns(2)
        with colC:
            region_fig = px.bar(
                region_dist,
                x="duration_hours",
                y="region_name",
                orientation="h",
                color="region_name",
                text="duration_label",
                title="Region-Wise Time Spent",
            )
            region_fig.update_layout(
                xaxis_title="Total Hours",
                yaxis_title="Region",
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            region_fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(region_fig, use_container_width=True)

        with colD:
            heatmap_fig = px.density_heatmap(
                heatmap_df,
                x="hour_of_day",
                y="region_name",
                z="active_persons",
                histfunc="sum",
                color_continuous_scale="YlOrRd",
                title="Hour x Region Activity Heatmap",
            )
            heatmap_fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Region",
                coloraxis_colorbar_title="Persons",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

        person_dist = (
            overview_df.groupby("person_label", as_index=False)["duration_sec"]
            .sum()
            .sort_values("duration_sec", ascending=False)
            .head(10)
        )
        person_dist["duration_hours"] = person_dist["duration_sec"] / 3600.0
        person_dist["duration_label"] = person_dist["duration_sec"].apply(_format_duration_hm)

        region_activity_df = (
            overview_df.groupby(["region_name", "activity"], as_index=False)["duration_sec"]
            .sum()
        )
        region_activity_df["duration_hours"] = region_activity_df["duration_sec"] / 3600.0

        colE, colF = st.columns(2)
        with colE:
            top_persons_fig = px.bar(
                person_dist,
                x="duration_hours",
                y="person_label",
                orientation="h",
                color="person_label",
                text="duration_label",
                title="Top Active Persons",
            )
            top_persons_fig.update_layout(
                xaxis_title="Total Hours",
                yaxis_title="Person",
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            top_persons_fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(top_persons_fig, use_container_width=True)

        with colF:
            region_activity_fig = px.bar(
                region_activity_df,
                x="region_name",
                y="duration_hours",
                color="activity",
                title="Region Activity Composition",
            )
            region_activity_fig.update_layout(
                xaxis_title="Region",
                yaxis_title="Total Hours",
                legend_title_text="Activity",
                barmode="stack",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(region_activity_fig, use_container_width=True)


class FaceRegistrationPage(BasePage):
    TARGET_POSES = ["Frontal", "Left Profile", "Right Profile", "Up Tilt", "Down Tilt"]
    STABLE_POLLS_REQUIRED = 2
    REFRESH_INTERVAL_MS = 700

    def __init__(self, service):
        super().__init__("🧠 Face Registration", service)

    @staticmethod
    def _capture_state(username: str = ""):
        state = st.session_state.get("face_guided_capture_state")
        if state is None:
            state = {
                "username": username,
                "captured_samples": {},
                "stable_counter": 0,
                "baseline_vertical_ratio": None,
                "baseline_yaw": None,
                "baseline_pitch": None,
                "status_message": "",
                "ignore_saved_samples": False,
            }
            st.session_state["face_guided_capture_state"] = state
        return state

    @staticmethod
    def _reset_capture_state(username: str = ""):
        st.session_state["face_guided_capture_state"] = {
            "username": username,
            "captured_samples": {},
            "stable_counter": 0,
            "baseline_vertical_ratio": None,
            "baseline_yaw": None,
            "baseline_pitch": None,
            "status_message": "",
            "ignore_saved_samples": True,
        }
        return st.session_state["face_guided_capture_state"]

    def _render_register_view(self):
        st.caption(
            "Client browser camera with threaded pose processing and auto-capture."
        )
        if webrtc_streamer is None or WebRtcMode is None or av is None:
            st.error("`streamlit-webrtc` is required for live client-side capture.")
            st.caption(
                "Install in the active environment: `pip install streamlit-webrtc`"
            )
            return

        if mp is None or _load_mediapipe_face_mesh() is None:
            st.error("MediaPipe Face Mesh is not available.")
            return

        current_url = str(getattr(st.context, "url", "") or "").strip()
        if not _is_secure_camera_context():
            st.error(
                "Camera access is blocked for this page because the app is opened from a non-secure URL."
            )
            if current_url:
                st.caption(f"Current URL: `{current_url}`")
            st.caption(
                "Use `http://localhost:8501` on the same machine, or use HTTPS for LAN access."
            )
            return

        top_cols = st.columns([4, 1.2])
        username = top_cols[0].text_input(
            label="Person Name",
            placeholder="Person Name",
            label_visibility="collapsed",
        )
        username = username.strip()
        if not username:
            self._reset_capture_state("")
            st.info("Enter a person name, then click Start to open the client camera.")
            return

        state = self._capture_state(username)
        if state.get("username") != username:
            state = self._reset_capture_state(username)
            state["ignore_saved_samples"] = False

        pose_samples_df = self.service.get_latest_pose_samples()
        reset_requested = top_cols[1].button("Reset Capture", width="stretch")
        if reset_requested:
            state = self._reset_capture_state(username)

        if not pose_samples_df.empty and not state.get("ignore_saved_samples", False):
            rows = pose_samples_df[
                pose_samples_df["username"].astype(str).str.strip() == username
            ]
            for _, row in rows.iterrows():
                pose = str(row.get("pose", "")).strip()
                image = row.get("face_image")
                if pose and isinstance(image, (bytes, bytearray)) and image:
                    state["captured_samples"][pose] = image

        remaining_poses = [
            pose for pose in self.TARGET_POSES if pose not in state["captured_samples"]
        ]
        target_pose = remaining_poses[0] if remaining_poses else "Done"

        st.markdown(
            """
            <style>
            .st-key-face_guided_capture_webrtc video,
            .st-key-face_guided_capture_webrtc canvas {
                width: 720px !important;
                max-width: 100% !important;
                height: 640px !important;
                max-height: 640px !important;
                object-fit: cover;
                border-radius: 10px;
                background: #020617;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        stream_col, crop_col = st.columns([2.4, 1.2])
        # Browser-to-app capture works with host ICE candidates here; avoiding
        # external STUN also prevents background retries during teardown.
        rtc_config = {"iceServers": []}
        should_stream = target_pose != "Done"
        with stream_col:
            with st.container(border=True):
                st.caption("Live Input Stream")
                ctx = webrtc_streamer(
                    key="face_guided_capture_webrtc",
                    mode=WebRtcMode.SENDRECV,
                    media_stream_constraints={"video": True, "audio": False},
                    rtc_configuration=rtc_config,
                    desired_playing_state=should_stream,
                    async_processing=True,
                    video_processor_factory=ThreadedFaceCaptureProcessor,
                )

        stream_error = ""
        if ctx is not None and hasattr(ctx, "state") and hasattr(ctx.state, "signalling"):
            stream_error = str(getattr(ctx.state, "signalling", "") or "").strip()

        if not should_stream:
            st.success("Face captured successfully. Camera closed.")
            if state.get("status_message"):
                st.caption(state["status_message"])
            with st.container(border=True, height=485):
                st.progress(
                    len(state["captured_samples"]) / float(len(self.TARGET_POSES))
                )
                for pose_name in self.TARGET_POSES:
                    st.caption(pose_name)
                    pose_img = state["captured_samples"].get(pose_name)
                    if pose_img:
                        st.image(pose_img, width="stretch")
                    else:
                        st.caption("Not captured")
            return

        if "navigator.mediadevices is undefined" in stream_error.lower():
            st.error(
                "Camera access is blocked because this page is not running in a secure browser context."
            )
            st.caption(
                "Open the app with `http://localhost:8501` on the same machine, or serve it over HTTPS if you need LAN access."
            )
            return

        if not ctx.state.playing:
            st.info("Click Start above to use this browser's camera.")
            return

        if st_autorefresh is not None:
            st_autorefresh(
                interval=self.REFRESH_INTERVAL_MS,
                key="face_guided_capture_refresh_active",
            )

        processor = ctx.video_processor
        if processor is None:
            st.info("Waiting for camera stream to initialize.")
            return

        processor.update_guidance(
            target_pose,
            state["baseline_vertical_ratio"],
            state["baseline_yaw"],
            state["baseline_pitch"],
        )
        live_state = processor.get_state()
        if live_state.get("error"):
            st.error(live_state["error"])
            return

        current_pose = live_state.get("current_pose", "No Face")
        current_direction = live_state.get("current_direction", "No Face")
        face_bytes = live_state.get("face_bytes")
        face_preview = live_state.get("face_preview")
        vertical_ratio = live_state.get("vertical_ratio")
        yaw_value = live_state.get("yaw")
        pitch_value = live_state.get("pitch")
        aligned = bool(live_state.get("aligned"))
        face_centered = bool(live_state.get("face_centered"))
        guide_message = live_state.get("guide_message", "")

        pose_match = _pose_matches_target(target_pose, current_pose, current_direction)
        ready_to_capture = pose_match and face_centered and face_bytes
        if ready_to_capture:
            state["stable_counter"] += 1
        else:
            state["stable_counter"] = 0

        info_cols = st.columns([1, 1, 1, 2])
        info_cols[0].metric("Target", target_pose)
        info_cols[1].metric("Detected", current_pose)
        info_cols[2].metric(
            "Progress", f"{state['stable_counter']}/{self.STABLE_POLLS_REQUIRED}"
        )
        status_parts = [
            "Centered" if face_centered else "Not centered",
            "Aligned" if aligned else "Size/guide not ideal",
            "Pose matched" if pose_match else "Pose not matched",
        ]
        info_cols[3].caption(guide_message)
        st.caption(" | ".join(status_parts))
        if target_pose == "Frontal" and state.get("baseline_yaw") is None:
            st.caption(
                "Initial calibration is based on your frontal capture. Hold a centered, straight pose."
            )

        if state["stable_counter"] >= self.STABLE_POLLS_REQUIRED:
            ok, message = self.service.save_face_samples(
                username,
                [
                    {
                        "face_image": face_bytes,
                        "pose": target_pose,
                        "lighting": "",
                        "glasses": False,
                    }
                ],
            )
            state["status_message"] = message
            state["stable_counter"] = 0
            if ok:
                state["captured_samples"][target_pose] = face_bytes
                if target_pose == "Frontal" and vertical_ratio is not None:
                    state["baseline_vertical_ratio"] = vertical_ratio
                if target_pose == "Frontal" and yaw_value is not None:
                    state["baseline_yaw"] = yaw_value
                if target_pose == "Frontal" and pitch_value is not None:
                    state["baseline_pitch"] = pitch_value
                st.rerun()

        if state.get("status_message"):
            st.caption(state["status_message"])

        with crop_col:
            with st.container(border=True):
                pass

        st.progress(len(state["captured_samples"]) / float(len(self.TARGET_POSES)))
        st.markdown("#### Captured Faces")
        cols = st.columns(len(self.TARGET_POSES))
        for i, pose_name in enumerate(self.TARGET_POSES):
            with cols[i]:
                st.caption(pose_name)
                pose_img = state["captured_samples"].get(pose_name)
                if pose_img:
                    st.image(pose_img, width="stretch")
                else:
                    st.caption("Not captured")

    def _render_manage_view(self):
        st.caption("Search, edit, and delete saved faces.")
        status_message = st.session_state.pop("face_manage_status_message", None)
        status_level = st.session_state.pop("face_manage_status_level", "success")
        if status_message:
            getattr(st, status_level, st.info)(status_message)

        faces_df = self.service.get_faces()
        sample_counts = self.service.get_face_sample_counts()
        pose_samples_df = self.service.get_latest_pose_samples()
        if faces_df.empty and sample_counts.empty:
            st.info("No saved faces yet.")
            return

        preview_df = faces_df.copy().sort_values("created_at", ascending=False)
        if not sample_counts.empty:
            preview_df = preview_df.merge(
                sample_counts[["username", "sample_count"]], on="username", how="left"
            )
            preview_df["sample_count"] = (
                preview_df["sample_count"].fillna(0).astype(int)
            )
        else:
            preview_df["sample_count"] = 0

        search_query = st.text_input(
            "Search Faces",
            value=st.session_state.get("manage_faces_search_query", ""),
            placeholder="Search by registered name",
            key="manage_faces_search_query",
        ).strip()
        if search_query:
            preview_df = preview_df[
                preview_df["username"]
                .astype(str)
                .str.contains(search_query, case=False, na=False)
            ]
            if preview_df.empty:
                st.info(f"No registered faces matched '{search_query}'.")
                return

        pose_order = [
            "Frontal",
            "Left Profile",
            "Right Profile",
            "Up Tilt",
            "Down Tilt",
        ]
        pose_map = {}
        if not pose_samples_df.empty:
            for _, p_row in pose_samples_df.iterrows():
                uname = str(p_row.get("username", "")).strip()
                pose = str(p_row.get("pose", "")).strip()
                img_blob = p_row.get("face_image")
                if not uname or not pose:
                    continue
                pose_map.setdefault(uname, {})[pose] = img_blob

        for _, row in preview_df.iterrows():
            username = str(row.get("username", "Unknown"))
            info_col, poses_col, action_col = st.columns([1.7, 5.3, 1.4])
            with info_col:
                st.markdown(f"**{username}**")
                st.caption(f"Saved: {row.get('created_at', '-')}")
                st.caption(f"Samples: {int(row.get('sample_count', 0))}")
            with poses_col:
                user_pose_samples = pose_map.get(username, {})
                p_cols = st.columns(5)
                for i, pose_name in enumerate(pose_order):
                    with p_cols[i]:
                        st.caption(pose_name)
                        pose_img = user_pose_samples.get(pose_name)
                        if isinstance(pose_img, (bytes, bytearray)) and pose_img:
                            st.image(pose_img, width="stretch")
                        else:
                            st.caption("Not captured")
            with action_col:
                if st.button("Edit Name", key=f"edit_face_{username}", width="stretch"):
                    st.session_state["face_name_edit_target"] = username
                    st.rerun()
                if st.button("Delete", key=f"delete_face_{username}", width="stretch"):
                    ok, message = self.service.delete_face(username)
                    if ok:
                        st.session_state["face_manage_status_message"] = message
                        st.session_state["face_manage_status_level"] = "success"
                        st.rerun()
                    else:
                        st.error(message)
            st.divider()

        edit_target = st.session_state.get("face_name_edit_target")
        if edit_target:
            self._render_edit_name_dialog(edit_target)

    @st.dialog("Edit Face Name")
    def _render_edit_name_dialog(self, username: str):
        st.caption(f"Update the registered name for {username}.")
        with st.form(f"edit_face_name_form_{username}", clear_on_submit=False):
            new_name = st.text_input("New Name", value=username)
            form_cols = st.columns(2)
            submitted = form_cols[0].form_submit_button("Save", width="stretch")
            canceled = form_cols[1].form_submit_button("Cancel", width="stretch")

        if submitted:
            ok, message = self.service.rename_face(username, new_name)
            if ok:
                st.session_state.pop("face_name_edit_target", None)
                st.session_state["face_manage_status_message"] = message
                st.session_state["face_manage_status_level"] = "success"
                st.rerun()
            st.error(message)

        if canceled:
            st.session_state.pop("face_name_edit_target", None)
            st.rerun()

    def render(self):
        self.show_title()
        mode = st.segmented_control(
            "Face Registration Mode",
            options=["Register Face", "Manage Faces"],
            default="Register Face",
            key="face_registration_mode",
        )

        if mode == "Register Face":
            self._render_register_view()
        else:
            self._render_manage_view()


class PersonAnalyticsPage(BasePage):
    def __init__(self, service):
        super().__init__("👤 Person Analytics", service)

    @st.dialog("Delete Selected Records")
    def _render_delete_records_dialog(self, selected_name: str, record_ids):
        ids = [int(rid) for rid in record_ids]
        st.warning(f"Delete {len(ids)} selected record(s) for {selected_name}?")
        with st.form(
            f"delete_person_records_form_{selected_name}", clear_on_submit=False
        ):
            col1, col2 = st.columns(2)
            confirm = col1.form_submit_button("Delete", type="primary", width="stretch")
            cancel = col2.form_submit_button("Cancel", width="stretch")

        if confirm:
            ok, message = self.service.delete_activity_records(ids)
            st.session_state.pop("person_delete_target_ids", None)
            st.session_state.pop("person_delete_target_name", None)
            st.session_state["person_delete_status"] = (
                "success" if ok else "error",
                message,
            )
            st.rerun()

        if cancel:
            st.session_state.pop("person_delete_target_ids", None)
            st.session_state.pop("person_delete_target_name", None)
            st.rerun()

    def render(self):
        self.show_title()
        _, logs_df = self.service.load_data()

        if logs_df.empty:
            st.info("No person analytics data available.")
            return

        if "name" not in logs_df.columns:
            st.error("Missing 'name' column in activity logs.")
            return

        invalid_names = {"", "unknown", "n/a", "none", "null", "nan"}
        person_series = logs_df["name"].dropna().astype(str).str.strip()
        valid_person_series = person_series[
            ~person_series.str.lower().isin(invalid_names)
        ].copy()
        if valid_person_series.empty:
            st.info("No named persons available in logs.")
            return

        display_name_by_key = {}
        for name in valid_person_series:
            key = name.lower()
            current = display_name_by_key.get(key)
            if current is None or (name[:1].isupper() and not current[:1].isupper()):
                display_name_by_key[key] = name

        persons = sorted(display_name_by_key.values(), key=lambda value: value.lower())

        st.markdown("#### Person Filters")
        filter_col0, filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(
            [1.6, 1, 1, 1, 1]
        )
        selected_name = filter_col0.selectbox(
            "Select Person", persons, key="person_select"
        )

        selected_name_key = selected_name.strip().lower()
        selected_person_df = logs_df[
            logs_df["name"].fillna("").astype(str).str.strip().str.lower()
            == selected_name_key
        ]
        min_dt = selected_person_df["start_datetime"].min()
        max_dt = selected_person_df["end_datetime"].max()
        min_date, max_selectable_date, default_date = _date_filter_bounds(min_dt, max_dt)

        start_date = filter_col1.date_input(
            "From Date",
            value=default_date,
            min_value=min_date,
            max_value=max_selectable_date,
            key="person_from_date",
        )
        end_date = filter_col2.date_input(
            "To Date",
            value=default_date,
            min_value=min_date,
            max_value=max_selectable_date,
            key="person_to_date",
        )
        start_time = filter_col3.time_input(
            "From Time", value=dt_time(0, 0), key="person_from_time"
        )
        end_time = filter_col4.time_input(
            "To Time", value=dt_time(23, 59, 59), key="person_to_time"
        )

        df = logs_df[
            logs_df["name"].fillna("").astype(str).str.strip().str.lower()
            == selected_name_key
        ].copy()
        if df.empty:
            st.info("No records found for the selected person.")
            return

        st.markdown(f"### {selected_name}")

        start_ts = pd.Timestamp(datetime.combine(start_date, start_time))
        end_ts = pd.Timestamp(datetime.combine(end_date, end_time))
        if start_ts > end_ts:
            st.warning("From datetime cannot be after To datetime.")
            return

        df = df[
            (df["start_datetime"] >= start_ts) & (df["start_datetime"] <= end_ts)
        ].copy()
        df = df[df["duration_sec"] > 1].copy()
        if df.empty:
            st.info("No records found for this person in the selected time period.")
            return

        status_payload = st.session_state.pop("person_delete_status", None)
        if status_payload:
            level, message = status_payload
            if level == "success":
                st.success(message)
            else:
                st.error(message)

        section = st.segmented_control(
            "Person Analytics Section",
            options=["Overview", "Activity Records"],
            default="Activity Records",
            key="person_analytics_section",
        )

        if section == "Overview":
            refresh_col1, refresh_col2 = st.columns([1.2, 4.8])
            with refresh_col1:
                live_update = st.toggle(
                    "Live Update",
                    value=True,
                    key="person_overview_live_update",
                )
            with refresh_col2:
                if live_update:
                    st.caption(
                        "Overview refreshes automatically every 5 seconds to pick up new database records."
                    )
                    if st_autorefresh is not None:
                        st_autorefresh(
                            interval=5000,
                            key=f"person_overview_refresh_{selected_name}",
                        )
            self._render_overview_section(df, selected_name, start_ts, end_ts)
        else:
            self._render_activity_records_section(df, selected_name)

    def _render_overview_section(self, df, selected_name, start_ts, end_ts):
        no_detection_label = f"{selected_name} not detected"

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Hours", f"{df['duration_sec'].sum() / 3600:.2f}")
        col2.metric("Activities", df["activity"].nunique())
        col3.metric("Regions", df["region_name"].nunique())

        col4, col5, col6 = st.columns(3)
        col4.metric("Total Records", len(df))
        top_activity = df["activity"].value_counts().idxmax() if not df.empty else "N/A"
        col5.metric("Top Activity", top_activity)
        top_region = (
            df["region_name"].value_counts().idxmax() if not df.empty else "N/A"
        )
        col6.metric("Most Visited", top_region)

        timeline_df = df.copy()
        timeline_df["activity"] = timeline_df["activity"].apply(_canonical_activity_label)
        timeline_df = timeline_df[timeline_df["activity"].notna()].copy()
        if timeline_df.empty:
            st.info(
                "No person activity matches the selected categories: Working, Not Working, Using Mobile, Sleeping, Head Down."
            )
            return
        timeline_df["region_name"] = (
            timeline_df["region_name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
        timeline_df["duration_hms"] = (
            pd.to_timedelta(timeline_df["duration_sec"].clip(lower=0), unit="s")
            .astype(str)
            .str.replace("0 days ", "", regex=False)
        )
        timeline_df = timeline_df.sort_values("start_datetime", ascending=True).reset_index(
            drop=True
        )

        gap_rows = []
        detected_window_start = pd.Timestamp(timeline_df["start_datetime"].min())
        detected_window_end = pd.Timestamp(timeline_df["end_datetime"].max())
        cursor = detected_window_start
        for row in timeline_df.itertuples(index=False):
            row_start = pd.Timestamp(row.start_datetime)
            row_end = pd.Timestamp(row.end_datetime)
            if row_start > cursor:
                gap_duration = max((row_start - cursor).total_seconds(), 0.0)
                if gap_duration > 0:
                    gap_rows.append(
                        {
                            "start_datetime": cursor,
                            "end_datetime": row_start,
                            "activity": no_detection_label,
                            "region_name": "No detection",
                            "duration_sec": gap_duration,
                            "duration_hms": str(pd.to_timedelta(gap_duration, unit="s")).replace(
                                "0 days ", ""
                            ),
                        }
                    )
            if row_end > cursor:
                cursor = row_end

        if gap_rows:
            timeline_df = pd.concat([timeline_df, pd.DataFrame(gap_rows)], ignore_index=True)

        timeline_df = timeline_df.sort_values("start_datetime", ascending=True).reset_index(
            drop=True
        )
        timeline_df["timeline_row"] = timeline_df["activity"]

        st.markdown("#### Activity Timeline")
        st.caption(
            f"Timeline for {selected_name} within the currently selected date and time range."
        )

        timeline_fig = px.timeline(
            timeline_df,
            x_start="start_datetime",
            x_end="end_datetime",
            y="timeline_row",
            color="activity",
            custom_data=["region_name", "duration_hms", "activity"],
            color_discrete_map={no_detection_label: "#6b7280"},
        )
        timeline_fig.update_yaxes(
            autorange="reversed",
            title_text="Activity",
            categoryorder="array",
            categoryarray=list(dict.fromkeys(timeline_df["timeline_row"].tolist())),
        )
        timeline_fig.update_xaxes(
            title_text="Time",
            range=[detected_window_start, detected_window_end],
        )
        timeline_fig.update_traces(
            hovertemplate=(
                "Activity: %{customdata[2]}<br>"
                "Region: %{customdata[0]}<br>"
                "Start: %{base|%Y-%m-%d %H:%M:%S}<br>"
                "End: %{x|%Y-%m-%d %H:%M:%S}<br>"
                "Duration: %{customdata[1]}<extra></extra>"
            )
        )
        timeline_fig.update_layout(
            height=max(360, 120 + 56 * timeline_df["timeline_row"].nunique()),
            legend_title_text="Activity",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        st.markdown("#### Activity Duration Summary")
        activity_summary_df = (
            timeline_df[timeline_df["activity"] != no_detection_label]
            .groupby("activity", as_index=False)["duration_sec"]
            .sum()
            .sort_values("duration_sec", ascending=False)
        )
        activity_summary_df["duration_hours"] = activity_summary_df["duration_sec"] / 3600.0
        activity_summary_df["duration_label"] = activity_summary_df["duration_sec"].apply(
            lambda secs: _format_duration_hm(secs)
        )

        activity_bar_fig = px.bar(
            activity_summary_df,
            x="activity",
            y="duration_hours",
            color="activity",
            text="duration_label",
            custom_data=["duration_label"],
        )
        activity_bar_fig.update_traces(
            hovertemplate=(
                "Activity: %{x}<br>"
                "Total Time: %{customdata[0]}<extra></extra>"
            )
        )
        activity_bar_fig.update_layout(
            xaxis_title="Activity",
            yaxis_title="Total Hours",
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(activity_bar_fig, use_container_width=True)

    def _render_activity_records_section(self, df, selected_name):
        if "id" not in df.columns:
            st.warning("Record IDs are not available, so delete is disabled.")
            return

        df = _exclude_analyzing(df)
        if df.empty:
            st.info("No activity records are available after excluding analyzing entries.")
            return

        df = df.sort_values("start_datetime", ascending=False).reset_index(drop=True)
        df["activity_filter_labels"] = df["activity"].apply(_activity_filter_labels)
        df["screenshot_path"] = _attach_activity_screenshot_paths(df)
        df["screenshot_thumb"] = df["screenshot_path"].apply(
            lambda p: _thumbnail_data_uri(p, max_side=300)
        )
        duration_total_sec = df["duration_sec"].round().astype("int64").clip(lower=0)
        duration_hours = (duration_total_sec // 3600).astype("int64")
        duration_minutes = ((duration_total_sec % 3600) // 60).astype("int64")
        duration_seconds = (duration_total_sec % 60).astype("int64")
        df["duration_hms"] = (
            duration_hours.map(lambda v: f"{int(v):02d}")
            + ":"
            + duration_minutes.map(lambda v: f"{int(v):02d}")
            + ":"
            + duration_seconds.map(lambda v: f"{int(v):02d}")
        )

        activity_options = sorted(
            {
                label
                for labels in df["activity_filter_labels"].tolist()
                for label in labels
            }
        )
        filter_col1, filter_col2 = st.columns([2.2, 1])
        with filter_col1:
            selected_activity_filters = st.multiselect(
                "Filter by Activity",
                options=activity_options,
                default=[],
                key=f"person_activity_filter_{selected_name}",
                placeholder="All activities",
            )
        with filter_col2:
            min_duration_sec = st.number_input(
                "Ignore Shorter Than (sec)",
                min_value=0,
                max_value=3600,
                value=3,
                step=1,
                key=f"person_min_duration_filter_{selected_name}",
            )

        if min_duration_sec > 0:
            df = df[df["duration_sec"] >= float(min_duration_sec)].copy()

        if df.empty:
            st.info("No activity records match the selected minimum duration.")
            return

        activity_options = sorted(
            {
                label
                for labels in df["activity_filter_labels"].tolist()
                for label in labels
            }
        )
        if selected_activity_filters:
            selected_activity_filter_set = set(selected_activity_filters)
            df = df[
                df["activity_filter_labels"].apply(
                    lambda labels: bool(selected_activity_filter_set.intersection(labels))
                )
            ].copy()

        st.caption(f"Showing {len(df)} record(s) for {selected_name}.")

        if df.empty:
            st.info("No activity records match the selected filters.")
            return

        record_ids = df["id"].tolist()
        checkbox_key_prefix = f"person_record_cb_{selected_name}"

        col_select_all, col_delete = st.columns([1, 5])

        person_status_list = []
        for idx, row in df.iterrows():
            start_dt = row.get("start_datetime")
            end_dt = row.get("end_datetime")
            start_time = (
                pd.Timestamp(start_dt).strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(start_dt)
                else "N/A"
            )
            end_time = (
                pd.Timestamp(end_dt).strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(end_dt)
                else "N/A"
            )
            duration = str(row.get("duration_hms", "") or "").strip()
            region_name = str(row.get("region_name", "")).strip() or "Unknown"
            activity = _format_person_status_activity(row.get("activity", ""))

            status_text = f"""⏱️ {start_time}
━━━━━━━━━━━━━━━━━
End: {end_time}
Duration: {duration}
━━━━━━━━━━━━━━━━━
📍 {region_name}
━━━━━━━━━━━━━━━━━
🎬 {activity}"""
            person_status_list.append(status_text)

        table_html = """
        <style>
          .person-records-head {
            background: #111827;
            color: #e5e7eb;
            font-family: "Trebuchet MS", "Verdana", sans-serif;
            font-size: 16px;
            font-weight: 700;
            padding: 14px 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.35);
          }
          .person-records-row {
            padding: 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
          }
          .person-records-row:last-child {
            border-bottom: none;
          }
          .person-records-row:hover {
            background: rgba(30, 41, 59, 0.45);
          }
          .person-status-region {
            color: #d1d5db;
            font-family: "Trebuchet MS", "Verdana", sans-serif;
            font-size: 16px;
            font-weight: 500;
            padding-top: 8px;
          }
          .person-status-text {
            color: #d1d5db;
            white-space: pre-wrap;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 13px;
            line-height: 1.25;
            word-break: break-word;
            padding-top: 2px;
          }
          .person-status-thumb {
            width: 220px;
            height: 160px;
            object-fit: contain;
            background: rgba(2, 6, 23, 0.55);
            display: block;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.4);
          }
          .person-status-no-thumb {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 220px;
            height: 160px;
            border-radius: 8px;
            border: 1px dashed rgba(148, 163, 184, 0.35);
            color: #9ca3af;
            font-size: 14px;
          }
          div[data-testid="stCheckbox"] label p {
            font-size: 0;
          }
        </style>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        all_selected = bool(record_ids) and all(
            bool(st.session_state.get(f"{checkbox_key_prefix}_{record_id}", False))
            for record_id in record_ids
        )
        toggle_button_label = (
            "Deselect All Records" if all_selected else "✓ Select All Records"
        )

        with col_select_all:
            toggle_all_clicked = st.button(
                toggle_button_label,
                key=f"{checkbox_key_prefix}_toggle_all_btn",
                use_container_width=True,
            )
        with col_delete:
            st.empty()

        if toggle_all_clicked:
            target_state = not all_selected
            for record_id in record_ids:
                st.session_state[f"{checkbox_key_prefix}_{record_id}"] = target_state
            st.rerun()

        selected_ids = []
        header_cols = st.columns([0.55, 2.4, 6.0, 3.0])
        with header_cols[0]:
            st.markdown('<div class="person-records-head">Sel</div>', unsafe_allow_html=True)
        with header_cols[1]:
            st.markdown('<div class="person-records-head">Region</div>', unsafe_allow_html=True)
        with header_cols[2]:
            st.markdown('<div class="person-records-head">Person Status</div>', unsafe_allow_html=True)
        with header_cols[3]:
            st.markdown('<div class="person-records-head">Screenshot</div>', unsafe_allow_html=True)

        records_container = st.container(height=720, border=False)
        with records_container:
            for record_id, region_name, status_text, screenshot_uri, screenshot_path, start_dt in zip(
                record_ids,
                df["region_name"].tolist(),
                person_status_list,
                df["screenshot_thumb"].tolist(),
                df["screenshot_path"].tolist(),
                df["start_datetime"].tolist(),
            ):
                cb_key = f"{checkbox_key_prefix}_{record_id}"
                row_cols = st.columns([0.55, 2.4, 6.0, 3.0], vertical_alignment="center")
                with row_cols[0]:
                    st.markdown('<div class="person-records-row">', unsafe_allow_html=True)
                    if st.checkbox(
                        f"Select Record {record_id}",
                        key=cb_key,
                        label_visibility="collapsed",
                    ):
                        selected_ids.append(record_id)
                    st.markdown("</div>", unsafe_allow_html=True)
                with row_cols[1]:
                    st.markdown(
                        f'<div class="person-records-row person-status-region">{html.escape(str(region_name))}</div>',
                        unsafe_allow_html=True,
                    )
                with row_cols[2]:
                    st.markdown(
                        f'<div class="person-records-row person-status-text">{html.escape(str(status_text))}</div>',
                        unsafe_allow_html=True,
                    )
                with row_cols[3]:
                    screenshot_uri = str(screenshot_uri or "").strip()
                    screenshot_path = str(screenshot_path or "").strip()
                    start_time_label = (
                        pd.Timestamp(start_dt).strftime("%Y-%m-%d %H:%M:%S")
                        if pd.notna(start_dt)
                        else "N/A"
                    )
                    st.markdown('<div class="person-records-row">', unsafe_allow_html=True)
                    _render_clickable_image_preview(
                        screenshot_uri,
                        _source_image_data_uri(screenshot_path),
                        f"{selected_name} • {region_name} • {start_time_label}",
                        frame_height=190,
                        image_width=220,
                        image_height=160,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

        selected_count = len(selected_ids)

        st.markdown("---")
        if selected_count > 0:
            st.warning(f"⚠️ {selected_count} record(s) selected for deletion")
            if st.button(
                f"🗑️ Confirm Delete ({selected_count})",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["person_delete_target_ids"] = selected_ids
                st.session_state["person_delete_target_name"] = selected_name
                st.rerun()


class RegionLiveSection(BaseSection):
    REFRESH_INTERVAL = 5

    def __init__(self, service):
        super().__init__(service)

    @st.fragment(run_every=REFRESH_INTERVAL)
    def _render_live_fragment(self):
        live_outputs = _collect_region_live_outputs()
        region_options = sorted(
            [
                {
                    "slug": region_slug,
                    "label": str(live_region.get("region_name") or region_slug).strip(),
                    "live": live_region,
                }
                for region_slug, live_region in live_outputs.items()
            ],
            key=lambda item: item["label"].lower(),
        )
        if not region_options:
            st.info(f"No live region outputs are available in {REGION_OUTPUT_DIR}.")
            return

        st.caption("Current region state updates independently from the logs section.")
        region_rows = []
        for region in region_options:
            live_state = region["live"].get("state", {})
            image_path = region["live"].get("image_path")
            image_src = (
                _thumbnail_data_uri(str(image_path), max_side=360)
                if image_path is not None
                else ""
            )
            full_image_src = (
                _source_image_data_uri(str(image_path))
                if image_path is not None
                else ""
            )

            region_rows.append(
                {
                    "region": region["label"],
                    "image_src": image_src,
                    "full_image_src": full_image_src,
                    "updated_at": str(live_state.get("updated_at") or "").strip(),
                    "occupancy": int(live_state.get("occupancy", 0) or 0),
                    "active": "Yes"
                    if bool(live_state.get("is_active", False))
                    else "No",
                    "people": live_state.get("people") or [],
                }
            )

        _render_region_live_horizontal_table(region_rows)

    def render(self):
        self._render_live_fragment()


class RegionLogsSection(BaseSection):
    def __init__(self, service):
        super().__init__(service)

    @staticmethod
    def _resolve_chunk_path(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        path = Path(text)
        if path.is_absolute():
            return str(path)
        return str(CVML_SURVEILLANCE_ROOT / path)

    def render(self):
        chunk_df = self.service.get_region_chunk_logs()
        if chunk_df.empty:
            st.info("No region chunk logs available.")
            return

        df = chunk_df.copy()
        if "chunk_start_datetime" in df.columns:
            df["start_datetime"] = pd.to_datetime(
                df["chunk_start_datetime"], errors="coerce"
            )
        else:
            df["start_datetime"] = pd.NaT
        if "chunk_end_datetime" in df.columns:
            df["end_datetime"] = pd.to_datetime(
                df["chunk_end_datetime"], errors="coerce"
            )
        else:
            df["end_datetime"] = pd.NaT

        df = df.dropna(subset=["start_datetime", "end_datetime"]).copy()
        if df.empty:
            st.info("Region chunk logs are missing valid timestamps.")
            return

        st.caption("Search and review region chunk snapshots and aggregated records.")
        df = _apply_time_period_filter(df, "region_chunk_logs")

        region_names = sorted(
            [
                str(v).strip()
                for v in df.get("region_name", pd.Series(dtype=str)).dropna().unique()
                if str(v).strip()
            ],
            key=lambda s: s.lower(),
        )

        st.markdown(
            """
            <style>
              /* Keep multiselect chips/tags in a single row (reduces vertical wrapping). */
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

        col1, col2 = st.columns([2.3, 1])
        with col1:
            selected_regions = st.multiselect(
                "Regions",
                options=region_names,
                default=region_names[:3] if len(region_names) > 3 else region_names,
                key="region_chunk_region_filter",
            )
        with col2:
            search_text = st.text_input(
                "Search (names / activities)", value="", key="region_chunk_search"
            )

        if selected_regions:
            df = df[
                df["region_name"].astype(str).isin([str(v) for v in selected_regions])
            ].copy()

        if search_text.strip():
            needle = search_text.strip().lower()
            names_text = (
                df.get("unique_names_json", pd.Series([""] * len(df), index=df.index))
                .astype(str)
                .str.lower()
            )
            activities_text = (
                df.get(
                    "activity_counts_json", pd.Series([""] * len(df), index=df.index)
                )
                .astype(str)
                .str.lower()
            )
            df = df[
                names_text.str.contains(needle, na=False)
                | activities_text.str.contains(needle, na=False)
            ].copy()

        if df.empty:
            st.info("No matching region chunk logs for the selected filters.")
            return

        df = df.sort_values("start_datetime", ascending=False).reset_index(drop=True)
        df["screenshot_thumb"] = df.get(
            "screenshot_path", pd.Series([""] * len(df), index=df.index)
        ).apply(
            lambda p: _thumbnail_data_uri(self._resolve_chunk_path(p), max_side=220)
        )
        df["screenshot_full"] = df.get(
            "screenshot_path", pd.Series([""] * len(df), index=df.index)
        ).apply(lambda p: _source_image_data_uri(self._resolve_chunk_path(p)))
        duration_sec = (
            (df["end_datetime"] - df["start_datetime"])
            .dt.total_seconds()
            .fillna(0.0)
            .clip(lower=0.0)
        )
        df["duration_hm"] = duration_sec.apply(_format_duration_hm)
        df["people"] = df.get("unique_names_json", "").apply(
            lambda v: _format_json_list(v, limit=10)
        )
        df["activities"] = df.get("activity_counts_json", "").apply(
            lambda v: _format_json_counts(v, limit=6)
        )
        df["poses"] = df.get("pose_counts_json", "").apply(
            lambda v: _format_json_counts(v, limit=6)
        )

        table_df = pd.DataFrame(
            {
                "Region": df.get("region_name", "").astype(str),
                "Start Time": df["start_datetime"],
                "End Time": df["end_datetime"],
                "Duration": df["duration_hm"],
                "Frames": df.get("frames_seen", 0),
                "Active Frames": df.get("active_frames", 0),
                "Avg Occupancy": df.get("avg_occupancy", 0.0),
                "Max Occupancy": df.get("max_occupancy", 0),
                "People": df["people"],
                "Activities": df["activities"],
                "Poses": df["poses"],
                "Screenshot": df["screenshot_thumb"],
                "Screenshot Full": df["screenshot_full"],
            }
        ).reset_index(drop=True)

        # Create Region Status column with organized information
        region_status_list = []
        for idx, row in table_df.iterrows():
            start_time = (
                row["Start Time"].strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(row["Start Time"])
                else "N/A"
            )
            end_time = (
                row["End Time"].strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(row["End Time"])
                else "N/A"
            )
            status_text = f"""⏱️ {start_time}
━━━━━━━━━━━━━━━━━
End: {end_time}
Duration: {row["Duration"]}
Occupancy: {row["Max Occupancy"]}
━━━━━━━━━━━━━━━━━
👥 {row["People"]}
━━━━━━━━━━━━━━━━━
📋 {row["Activities"]}"""
            region_status_list.append(status_text)

        rows_html = []
        for region_name, status_text, screenshot_uri, screenshot_full_uri in zip(
            table_df["Region"].tolist(),
            region_status_list,
            table_df["Screenshot"].tolist(),
            table_df["Screenshot Full"].tolist(),
        ):
            region_escaped = html.escape(str(region_name))
            status_html = html.escape(str(status_text)).replace("\n", "<br/>")

            screenshot_uri = str(screenshot_uri or "").strip()
            screenshot_full_uri = str(screenshot_full_uri or screenshot_uri).strip()
            screenshot_cell = (
                (
                    '<button type="button" class="region-status-thumb-button" '
                    f'data-title="{region_escaped}" '
                    f'data-src="{html.escape(screenshot_full_uri, quote=True)}" '
                    'onclick="openRegionLogImage(this.dataset.title, this.dataset.src)">'
                    f'<img src="{screenshot_uri}" class="region-status-thumb" alt="snapshot" />'
                    '<span class="region-status-thumb-overlay">Click for full view</span>'
                    "</button>"
                )
                if screenshot_uri
                else '<div class="region-status-no-thumb">No image</div>'
            )

            rows_html.append(
                "<tr>"
                f'<td class="region-status-region">{region_escaped}</td>'
                f'<td class="region-status-text">{status_html}</td>'
                f'<td class="region-status-shot">{screenshot_cell}</td>'
                "</tr>"
            )

        body_html = "".join(rows_html)
        table_html = """
        <style>
          .region-logs-wrap {
            max-height: 760px;
            overflow-y: auto;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
          }
          .region-logs-table {
            width: 100%;
            border-collapse: collapse;
            font-family: "Trebuchet MS", "Verdana", sans-serif;
            font-size: 16px;
          }
          .region-logs-table thead th {
            position: sticky;
            top: 0;
            z-index: 2;
            background: #111827;
            color: #e5e7eb;
            text-align: left;
            font-weight: 700;
            padding: 14px 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.35);
          }
          .region-logs-table td {
            color: #d1d5db;
            padding: 14px 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
            vertical-align: top;
          }
          .region-logs-table tbody tr:hover td {
            background: rgba(30, 41, 59, 0.45);
          }
          .region-status-region { width: 28%; }
          .region-status-text {
            white-space: normal;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 13px;
            line-height: 1.25;
            word-break: break-word;
          }
          .region-status-shot { width: 24%; }
          .region-status-thumb {
            width: 220px;
            height: 160px;
            object-fit: contain;
            background: rgba(2, 6, 23, 0.55);
            display: block;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.4);
          }
          .region-status-thumb-button {
            position: relative;
            display: inline-flex;
            padding: 0;
            border: none;
            background: transparent;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
          }
          .region-status-thumb-overlay {
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
          }
          .region-status-thumb-button:hover .region-status-thumb-overlay {
            opacity: 1;
          }
          .region-status-no-thumb {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 220px;
            height: 160px;
            border-radius: 8px;
            border: 1px dashed rgba(148, 163, 184, 0.35);
            color: #9ca3af;
            font-size: 14px;
          }
          .region-log-image-modal {
            position: fixed;
            inset: 0;
            display: none;
            align-items: center;
            justify-content: center;
            background: rgba(2, 6, 23, 0.96);
            z-index: 9999;
            padding: 8px;
            box-sizing: border-box;
          }
          .region-log-image-modal.is-open {
            display: flex;
          }
          .region-log-image-modal__content {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 14px;
          }
          .region-log-image-modal__topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
          }
          .region-log-image-modal__title {
            color: #e5e7eb;
            font-size: 28px;
            font-weight: 700;
          }
          .region-log-image-modal__back {
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(15, 23, 42, 0.88);
            color: #e5e7eb;
            border-radius: 10px;
            padding: 12px 20px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
          }
          .region-log-image-modal__viewport {
            flex: 1;
            min-height: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: rgba(15, 23, 42, 0.45);
          }
          .region-log-image-modal__img {
            max-width: 98vw;
            max-height: calc(100vh - 100px);
            object-fit: contain;
            display: block;
          }
        </style>
        <div class="region-logs-wrap">
          <table class="region-logs-table">
            <thead>
              <tr>
                <th>Region</th>
                <th>Region Status</th>
                <th>Screenshot</th>
              </tr>
            </thead>
            <tbody>
              __BODY__
            </tbody>
          </table>
        </div>
        <div id="region-log-image-modal" class="region-log-image-modal" onclick="closeRegionLogImage(event)">
          <div class="region-log-image-modal__content">
            <div class="region-log-image-modal__topbar">
              <div id="region-log-image-modal-title" class="region-log-image-modal__title">Region Snapshot</div>
              <button type="button" class="region-log-image-modal__back" onclick="closeRegionLogImage(event)">Back</button>
            </div>
            <div class="region-log-image-modal__viewport">
              <img id="region-log-image-modal-img" class="region-log-image-modal__img" alt="Region log full view" />
            </div>
          </div>
        </div>
        <script>
          function openRegionLogImage(title, imageSrc) {
            var modal = document.getElementById("region-log-image-modal");
            var modalTitle = document.getElementById("region-log-image-modal-title");
            var modalImage = document.getElementById("region-log-image-modal-img");
            if (!modal || !modalTitle || !modalImage) return;
            modalTitle.textContent = title || "Region Snapshot";
            modalImage.src = imageSrc || "";
            modal.classList.add("is-open");
            document.body.style.overflow = "hidden";
            if (modal.requestFullscreen) {
              modal.requestFullscreen().catch(function() {});
            }
          }
          function closeRegionLogImage(event) {
            if (event) {
              event.preventDefault();
              event.stopPropagation();
            }
            var modal = document.getElementById("region-log-image-modal");
            var modalImage = document.getElementById("region-log-image-modal-img");
            if (!modal || !modalImage) return;
            modal.classList.remove("is-open");
            modalImage.src = "";
            if (document.fullscreenElement) {
              document.exitFullscreen().catch(function() {});
            }
            document.body.style.overflow = "";
          }
          document.addEventListener("keydown", function(event) {
            if (event.key === "Escape") {
              closeRegionLogImage();
            }
          });
        </script>
        """
        components.html(table_html.replace("__BODY__", body_html), height=980, scrolling=False)

        csv = table_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Region Logs CSV", csv, "region_chunk_logs.csv")


class RegionAnalyticsPage(BasePage):
    def __init__(self, service):
        super().__init__("🗺️ Region Analytics", service)
        self.live_section = RegionLiveSection(service)
        self.logs_section = RegionLogsSection(service)

    def render(self):
        self.show_title()
        section = st.segmented_control(
            "Region Analytics Section",
            options=["Live View", "Region Logs"],
            default="Live View",
            key="region_analytics_section",
        )

        if section == "Live View":
            self.live_section.render()
        else:
            self.logs_section.render()


class TimelinePage(BasePage):
    def __init__(self, service):
        super().__init__("📅 Timeline", service)

    def render(self):
        self.show_title()
        _, logs_df = self.service.load_data()

        uid = st.selectbox("Select Person", sorted(logs_df["uid"].unique()))
        df = logs_df[logs_df["uid"] == uid]

        fig = px.timeline(
            df,
            x_start="start_datetime",
            x_end="end_datetime",
            y="activity",
            color="region_name",
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="content")


class RawLogsPage(BasePage):
    def __init__(self, service):
        super().__init__("📋 Raw Logs", service)

    def render(self):
        self.show_title()
        
        # Get total count first
        total_count = self.service.get_logs_count()
        
        # Create paginator
        paginator = PaginationManager('lab_raw_logs', total_count, default_per_page=50)
        
        # Load paginated data
        logs_df = self.service.get_logs(page=paginator.current_page, per_page=paginator.per_page)

        # Display data
        if not logs_df.empty:
            st.dataframe(
                logs_df.sort_values("start_datetime", ascending=False), width="content"
            )
        
        # Render pagination controls AFTER the table
        if paginator.render_pagination_controls():
            st.rerun()

        # Download button (downloads current page)
        if not logs_df.empty:
            csv = logs_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download CSV", csv, "activity_logs.csv")


class LiveCameraPage(BasePage):
    LIVE_IMAGE_PATH = (
        Path(LAB_CONFIG["live_image_path"]) if LAB_CONFIG.get("live_image_path") else None
    )
    LIVE_IMAGE_DIR = Path(
        LAB_CONFIG.get(
            "live_frames_dir",
            LIVE_IMAGE_PATH.parent if LIVE_IMAGE_PATH is not None else ".",
        )
    )
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    AUTO_REFRESH_SECONDS = 1

    def __init__(self, service):
        super().__init__("📷 Live Camera Snapshot", service)

    def _available_images(self):
        live_dir = self.LIVE_IMAGE_DIR
        if not live_dir.exists():
            return []

        files = [
            path
            for path in live_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    @staticmethod
    def _format_timestamp(path: Path):
        modified_at = datetime.fromtimestamp(path.stat().st_mtime)
        return modified_at.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _load_frame(path: Path, retries=3, retry_delay=0.05):
        for _ in range(retries):
            try:
                image_bytes = path.read_bytes()
            except FileNotFoundError:
                time.sleep(retry_delay)
                continue

            if not image_bytes:
                time.sleep(retry_delay)
                continue

            frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                time.sleep(retry_delay)
                continue

            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return None

    def render(self):
        self.show_title()
        if st_autorefresh is not None:
            st_autorefresh(
                interval=self.AUTO_REFRESH_SECONDS * 1000, key="lab_live_camera_refresh"
            )

        images = self._available_images()
        if not images:
            st.error(f"No camera snapshots found in: {self.LIVE_IMAGE_DIR}")
            return

        # Default to live_latest.jpg when present, otherwise the most recent file.
        default_image = (
            self.LIVE_IMAGE_PATH
            if self.LIVE_IMAGE_PATH is not None and self.LIVE_IMAGE_PATH.exists()
            else images[0]
        )
        default_index = images.index(default_image) if default_image in images else 0
        image_labels = {path: f"Camera {idx}" for idx, path in enumerate(images, start=1)}

        selected_image = st.selectbox(
            "Select Camera Snapshot",
            images,
            index=default_index,
            format_func=lambda path: image_labels.get(path, path.name),
        )

        st.caption(f"Last updated: {self._format_timestamp(selected_image)}")
        frame = self._load_frame(selected_image)
        if frame is None:
            st.info("Frame loading...")
        else:
            try:
                st.image(
                    frame,
                    caption=image_labels.get(selected_image, selected_image.name),
                    width="stretch",
                )
            except OSError:
                st.info("Frame loading...")
