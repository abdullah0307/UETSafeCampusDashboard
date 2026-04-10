import json
import threading
import time
from collections import deque
from statistics import median

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from lab_survelliance.utils import (
    _classify_direction_from_angles,
    _create_mediapipe_face_mesh,
    _direction_to_pose,
    _estimate_head_angles_from_mesh,
)


class PoseTunerProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=1)
        self._stop_event = threading.Event()
        self._mesh = _create_mediapipe_face_mesh()
        self._state = {
            "yaw": None,
            "pitch": None,
            "vertical_ratio": None,
            "direction": "No Face",
            "pose": "No Face",
            "error": "",
        }
        self._last_direction = "Forward-Level"
        self._last_yaw = None
        self._last_pitch = None
        self._angle_smooth = 0.75
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def get_state(self):
        with self._lock:
            return dict(self._state)

    def _loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                frame = self._queue.pop() if self._queue else None
                self._queue.clear()

            if frame is None:
                time.sleep(0.01)
                continue

            annotated = frame.copy()
            state = {
                "yaw": None,
                "pitch": None,
                "vertical_ratio": None,
                "direction": "No Face",
                "pose": "No Face",
                "error": "",
            }

            if self._mesh is None:
                state["error"] = "MediaPipe Face Mesh is not available."
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self._mesh.process(rgb)
                if result.multi_face_landmarks:
                    lms = result.multi_face_landmarks[0].landmark
                    yaw, pitch, vertical_ratio = _estimate_head_angles_from_mesh(
                        lms,
                        frame.shape[1],
                        frame.shape[0],
                    )
                    if yaw is not None and pitch is not None:
                        if self._last_yaw is not None:
                            yaw = self._angle_smooth * self._last_yaw + (1 - self._angle_smooth) * yaw
                            pitch = self._angle_smooth * self._last_pitch + (1 - self._angle_smooth) * pitch
                        self._last_yaw, self._last_pitch = yaw, pitch
                        direction = _classify_direction_from_angles(
                            yaw,
                            pitch,
                            self._last_direction.split("-")[1] if "-" in self._last_direction else "Level",
                            vertical_ratio=vertical_ratio,
                            baseline_vertical_ratio=None,
                        )
                        self._last_direction = direction
                        state = {
                            "yaw": float(yaw),
                            "pitch": float(pitch),
                            "vertical_ratio": float(vertical_ratio) if vertical_ratio is not None else None,
                            "direction": direction,
                            "pose": _direction_to_pose(direction),
                            "error": "",
                        }
                else:
                    self._last_direction = "Forward-Level"

            lines = [
                f"Pose: {state['pose']}",
                f"Direction: {state['direction']}",
                f"Yaw: {state['yaw']:.2f}" if state["yaw"] is not None else "Yaw: -",
                f"Pitch: {state['pitch']:.2f}" if state["pitch"] is not None else "Pitch: -",
                (
                    f"Vertical Ratio: {state['vertical_ratio']:.4f}"
                    if state["vertical_ratio"] is not None else "Vertical Ratio: -"
                ),
            ]
            if state["error"]:
                lines.append(state["error"])

            y = 32
            for line in lines:
                cv2.putText(
                    annotated,
                    line,
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 34

            with self._lock:
                self._state = state
                self._annotated = annotated

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        with self._lock:
            self._queue.append(image)
            annotated = getattr(self, "_annotated", None)
            output = annotated.copy() if isinstance(annotated, np.ndarray) else image
        return av.VideoFrame.from_ndarray(output, format="bgr24")


def _init_records():
    if "pose_tuner_records" not in st.session_state:
        st.session_state["pose_tuner_records"] = {
            "Frontal": [],
            "Left Profile": [],
            "Right Profile": [],
            "Up Tilt": [],
            "Down Tilt": [],
        }
    return st.session_state["pose_tuner_records"]


def _median_summary(records):
    if not records:
        return None
    return {
        "yaw": median([row["yaw"] for row in records]),
        "pitch": median([row["pitch"] for row in records]),
        "vertical_ratio": median([row["vertical_ratio"] for row in records]),
    }


def _classify_from_tuners(yaw, pitch, vertical_ratio, tuners):
    if yaw is None or pitch is None or vertical_ratio is None:
        return "No Face", "No Face"

    frontal_yaw_deadzone = tuners["frontal_yaw_deadzone"]
    frontal_pitch_deadzone = tuners["frontal_pitch_deadzone"]
    left_delta_threshold = tuners["left_delta_threshold"]
    right_delta_threshold = tuners["right_delta_threshold"]
    up_delta_threshold = tuners["up_delta_threshold"]
    down_delta_threshold = tuners["down_delta_threshold"]
    level_delta_deadzone = tuners["level_delta_deadzone"]
    baseline_yaw = tuners["baseline_yaw"]
    baseline_vertical_ratio = tuners["baseline_vertical_ratio"]

    yaw_delta = yaw - baseline_yaw
    vr_delta = vertical_ratio - baseline_vertical_ratio

    if abs(yaw_delta) <= frontal_yaw_deadzone and abs(pitch) <= frontal_pitch_deadzone:
        direction = "Forward-Level"
    else:
        if yaw_delta <= left_delta_threshold:
            horizontal = "Left"
        elif yaw_delta >= right_delta_threshold:
            horizontal = "Right"
        else:
            horizontal = "Forward"

        if vr_delta >= up_delta_threshold:
            vertical = "Up"
        elif vr_delta <= down_delta_threshold:
            vertical = "Down"
        elif abs(vr_delta) <= level_delta_deadzone:
            vertical = "Level"
        else:
            vertical = "Level"
        direction = f"{horizontal}-{vertical}"

    return direction, _direction_to_pose(direction)


st.set_page_config(layout="wide", page_title="Pose Tuner")
st.title("Pose Tuner")
st.caption("Record raw yaw, pitch, and vertical-ratio values, then tune thresholds with sliders.")

records = _init_records()
st_autorefresh(interval=700, key="pose_tuner_refresh")

with st.sidebar:
    st.subheader("Trackbar Tuning")
    st.caption("Adjust these sliders while watching the live classified pose.")
    frontal_yaw_deadzone = st.slider("Frontal Yaw Deadzone", 0.0, 40.0, 14.0, 1.0)
    frontal_pitch_deadzone = st.slider("Frontal Pitch Deadzone", 0.0, 40.0, 9.0, 1.0)
    baseline_yaw = st.slider("Baseline Yaw", -120.0, 120.0, 37.0, 1.0)
    baseline_vertical_ratio = st.slider("Baseline Vertical Ratio", 0.0, 1.0, 0.54, 0.005)
    left_delta_threshold = st.slider("Left Delta Threshold", -120.0, 0.0, -45.0, 1.0)
    right_delta_threshold = st.slider("Right Delta Threshold", 0.0, 120.0, 14.0, 1.0)
    up_delta_threshold = st.slider("Up Delta Threshold", 0.0, 0.5, 0.10, 0.005)
    down_delta_threshold = st.slider("Down Delta Threshold", -0.5, 0.0, -0.12, 0.005)
    level_delta_deadzone = st.slider("Level Delta Deadzone", 0.0, 0.2, 0.06, 0.005)

    tuner_values = {
        "frontal_yaw_deadzone": frontal_yaw_deadzone,
        "frontal_pitch_deadzone": frontal_pitch_deadzone,
        "baseline_yaw": baseline_yaw,
        "baseline_vertical_ratio": baseline_vertical_ratio,
        "left_delta_threshold": left_delta_threshold,
        "right_delta_threshold": right_delta_threshold,
        "up_delta_threshold": up_delta_threshold,
        "down_delta_threshold": down_delta_threshold,
        "level_delta_deadzone": level_delta_deadzone,
    }

target_pose = st.selectbox(
    "Pose To Record",
    ["Frontal", "Left Profile", "Right Profile", "Up Tilt", "Down Tilt"],
)

rtc_config = {"iceServers": []}
ctx = webrtc_streamer(
    key="pose_tuner_stream",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=rtc_config,
    async_processing=True,
    video_processor_factory=PoseTunerProcessor,
)

if not ctx.state.playing:
    st.info("Click Start above to open the client camera.")
    st.stop()

processor = ctx.video_processor
if processor is None:
    st.info("Waiting for stream initialization.")
    st.stop()

st_autorefresh = st.empty()
st_autorefresh.caption("Refreshes automatically while the stream is running.")
time.sleep(0.01)
state = processor.get_state()

metric_cols = st.columns(4)
metric_cols[0].metric("Detected Pose", state["pose"])
metric_cols[1].metric("Yaw", "-" if state["yaw"] is None else f"{state['yaw']:.2f}")
metric_cols[2].metric("Pitch", "-" if state["pitch"] is None else f"{state['pitch']:.2f}")
metric_cols[3].metric(
    "Vertical Ratio",
    "-" if state["vertical_ratio"] is None else f"{state['vertical_ratio']:.4f}",
)

custom_direction, custom_pose = _classify_from_tuners(
    state["yaw"],
    state["pitch"],
    state["vertical_ratio"],
    tuner_values,
)
custom_cols = st.columns(4)
custom_cols[0].metric("Custom Pose", custom_pose)
custom_cols[1].metric("Custom Direction", custom_direction)
custom_cols[2].metric(
    "Yaw Delta",
    "-" if state["yaw"] is None else f"{state['yaw'] - tuner_values['baseline_yaw']:.2f}",
)
custom_cols[3].metric(
    "VR Delta",
    "-" if state["vertical_ratio"] is None else f"{state['vertical_ratio'] - tuner_values['baseline_vertical_ratio']:.4f}",
)

action_cols = st.columns([1, 1, 3])
if action_cols[0].button("Record Current Reading", width='stretch'):
    if None not in (state["yaw"], state["pitch"], state["vertical_ratio"]):
        records[target_pose].append(
            {
                "yaw": round(state["yaw"], 4),
                "pitch": round(state["pitch"], 4),
                "vertical_ratio": round(state["vertical_ratio"], 6),
                "detected_pose": state["pose"],
                "direction": state["direction"],
            }
        )
        st.rerun()

if action_cols[1].button("Clear Records", width='stretch'):
    st.session_state["pose_tuner_records"] = {
        "Frontal": [],
        "Left Profile": [],
        "Right Profile": [],
        "Up Tilt": [],
        "Down Tilt": [],
    }
    st.rerun()

summary_cols = st.columns(5)
export_payload = {}
for idx, pose_name in enumerate(records.keys()):
    pose_records = records[pose_name]
    summary = _median_summary(pose_records)
    with summary_cols[idx]:
        st.caption(pose_name)
        st.caption(f"Samples: {len(pose_records)}")
        if summary is None:
            st.caption("No readings")
        else:
            st.caption(f"yaw: {summary['yaw']:.2f}")
            st.caption(f"pitch: {summary['pitch']:.2f}")
            st.caption(f"vr: {summary['vertical_ratio']:.4f}")
            export_payload[pose_name] = summary

st.markdown("#### Export Values")
st.code(json.dumps(export_payload, indent=2), language="json")
st.markdown("#### Export Tuner Settings")
st.code(json.dumps(tuner_values, indent=2), language="json")
st.caption("Record 5 to 10 readings for each pose, adjust the sliders, then share both JSON blocks.")
