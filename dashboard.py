from pathlib import Path
from collections import deque
import threading
import time
import sys

import altair as alt
import av
import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    ADAPTIVE_EAR_WINDOW,
    CAMERA_INDEX,
    EAR_SMOOTHING_WINDOW,
    EYE_CLOSED_EAR_THRESHOLD,
    HEAD_DOWN_PITCH_THRESHOLD,
    HEAD_TILT_ROLL_THRESHOLD,
    HEAD_UP_PITCH_THRESHOLD,
    ML_DROWSY_PROBABILITY_THRESHOLD,
    SIDE_LOOK_YAW_THRESHOLD,
    TELEMETRY_SAMPLE_SECONDS,
    YAWN_MAR_THRESHOLD,
)
from face_landmarks import create_face_landmarker  # noqa: E402
from live_interface import (  # noqa: E402
    analyze_face,
    unavailable_ml_result,
    update_no_face_telemetry,
)
from ml_predictor import load_ml_model  # noqa: E402
from telemetry_store import read_telemetry, reset_telemetry  # noqa: E402
from temporal_analysis import TemporalFatigueAnalyzer  # noqa: E402


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

VIDEO_CONSTRAINTS = {
    "video": {
        "width": {"ideal": 1280},
        "height": {"ideal": 720},
        "frameRate": {"ideal": 30, "max": 30},
    },
    "audio": False,
}


class SafeDriveVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.landmarker = None
        self.ml_model = None
        self.is_initialized = False
        self.initialization_error = None
        self.closed_start_time = None
        self.heavy_start_time = None
        self.last_sound_time = 0.0
        self.ear_history = deque(maxlen=EAR_SMOOTHING_WINDOW)
        self.adaptive_ear_history = deque(maxlen=ADAPTIVE_EAR_WINDOW)
        self.ear_threshold = EYE_CLOSED_EAR_THRESHOLD
        self.session_start_time = time.time()
        self.last_telemetry_sample_time = 0.0
        self.temporal_analyzer = TemporalFatigueAnalyzer()
        self.ml_result = unavailable_ml_result(None)
        self.last_timestamp_ms = 0
        self.latest = {
            "face_detected": False,
            "ml_prediction": "Initializing",
            "ml_drowsy_probability": None,
            "elapsed": 0.0,
            "error": None,
        }
        reset_telemetry()

    def _ensure_initialized(self):
        if self.is_initialized:
            return True

        try:
            self.landmarker = create_face_landmarker()
            self.ml_model = load_ml_model()
            self.ml_result = unavailable_ml_result(self.ml_model)
            self.is_initialized = True
            self.initialization_error = None
            return True
        except Exception as error:
            self.initialization_error = str(error)
            self._set_latest(False, time.time() - self.session_start_time, self.initialization_error)
            return False

    def _next_timestamp_ms(self):
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _set_latest(self, face_detected, elapsed, error=None):
        with self.lock:
            self.latest = {
                "face_detected": face_detected,
                "ml_prediction": self.ml_result.get("ml_prediction", "Unavailable"),
                "ml_drowsy_probability": self.ml_result.get("ml_drowsy_probability"),
                "ml_raw_drowsy_probability": self.ml_result.get("ml_raw_drowsy_probability"),
                "elapsed": elapsed,
                "error": error,
            }

    def get_latest(self):
        with self.lock:
            return dict(self.latest)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        if not self._ensure_initialized():
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        try:
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame),
                self._next_timestamp_ms(),
            )

            face_detected = bool(result.face_landmarks)
            face_landmarks = result.face_landmarks[0] if face_detected else None
            if face_detected:
                (
                    self.ml_result,
                    self.closed_start_time,
                    self.heavy_start_time,
                    self.last_sound_time,
                    self.ear_threshold,
                    self.last_telemetry_sample_time,
                ) = analyze_face(
                    image,
                    face_landmarks,
                    self.closed_start_time,
                    self.heavy_start_time,
                    self.last_sound_time,
                    self.ear_history,
                    self.adaptive_ear_history,
                    self.ear_threshold,
                    self.session_start_time,
                    self.last_telemetry_sample_time,
                    self.ml_model,
                    self.temporal_analyzer,
                )
            else:
                self.closed_start_time = None
                self.heavy_start_time = None
                self.ml_result = unavailable_ml_result(self.ml_model)
                elapsed = time.time() - self.session_start_time
                self.last_telemetry_sample_time = update_no_face_telemetry(
                    elapsed,
                    self.ear_threshold,
                    self.ml_model,
                    self.last_telemetry_sample_time,
                    self.temporal_analyzer,
                )

            elapsed = time.time() - self.session_start_time
            self._set_latest(face_detected, elapsed)
        except Exception as error:
            elapsed = time.time() - self.session_start_time
            self._set_latest(False, elapsed, str(error))

        return av.VideoFrame.from_ndarray(image, format="bgr24")


def safe_int(value, default=0):
    if value is None or pd.isna(value):
        return default

    return int(value)


def safe_float_text(value, digits=2):
    if value is None or pd.isna(value):
        return "-"

    return f"{float(value):.{digits}f}"


def safe_percent_text(value, digits=0):
    if value is None or pd.isna(value):
        return "-"

    return f"{float(value) * 100:.{digits}f}%"


def make_line_chart(df, value_columns, title, y_title, rules=None, height=250):
    if df.empty:
        return alt.Chart(pd.DataFrame({"time": [], "metric": [], "value": []})).mark_line()

    value_columns = [column for column in value_columns if column in df.columns]
    if not value_columns:
        return alt.Chart(pd.DataFrame({"time": [], "metric": [], "value": []})).mark_line()

    chart_data = df.melt(
        id_vars=["time"],
        value_vars=value_columns,
        var_name="metric",
        value_name="value",
    ).dropna()

    base = (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("time:Q", title="Seconds"),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("metric:N", title=None),
            tooltip=[
                alt.Tooltip("time:Q", format=".1f", title="Seconds"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", format=".3f", title="Value"),
            ],
        )
        .properties(title=title, height=height)
    )

    if not rules:
        return base

    layers = [base]
    for rule in rules:
        rule_df = pd.DataFrame({"value": [rule["value"]], "label": [rule["label"]]})
        layers.append(
            alt.Chart(rule_df)
            .mark_rule(strokeDash=[6, 4], color=rule.get("color", "#ef4444"), strokeWidth=2)
            .encode(y="value:Q")
        )
        layers.append(
            alt.Chart(rule_df)
            .mark_text(
                align="left",
                baseline="bottom",
                dx=6,
                dy=-2,
                color=rule.get("color", "#ef4444"),
                fontSize=11,
            )
            .encode(x=alt.value(8), y="value:Q", text="label:N")
        )

    return alt.layer(*layers).resolve_scale(color="independent")


def render_webrtc_camera():
    st.subheader("Live Camera")
    st.caption(
        "Click START and allow browser camera access. The WebRTC stream replaces the OpenCV camera window."
    )

    ctx = webrtc_streamer(
        key="safedrive-webrtc-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=VIDEO_CONSTRAINTS,
        video_processor_factory=SafeDriveVideoProcessor,
        async_processing=True,
        sendback_audio=False,
    )

    if ctx.video_processor is None:
        st.info("Waiting for the WebRTC camera stream to start.")
        return ctx

    latest = ctx.video_processor.get_latest()
    if latest.get("error"):
        st.error(f"Video processing error: {latest['error']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("ML fatigue score", safe_percent_text(latest.get("ml_drowsy_probability")))
    c2.metric("ML decision", latest.get("ml_prediction", "-"))
    c3.metric("Face detected", "Yes" if latest.get("face_detected") else "No")

    return ctx


def render_metric_cards(record):
    if record is None:
        st.info("Waiting for data from the WebRTC camera stream.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{safe_int(record.get('score'))}/100")
    c2.metric("EAR", safe_float_text(record.get("ear")))
    c3.metric("MAR", safe_float_text(record.get("mar")))
    c4.metric("Alert level", safe_int(record.get("alert_level")))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("EAR threshold", safe_float_text(record.get("ear_threshold")))
    c6.metric("Eye status", record.get("eye_status", "-"))
    c7.metric("Mouth status", record.get("mouth_status", "-"))
    c8.metric("Head status", record.get("head_status", "-"))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("ML decision", record.get("ml_prediction", "-"))
    c10.metric("Live drowsy probability", safe_percent_text(record.get("ml_drowsy_probability")))
    c11.metric("Raw model drowsy", safe_percent_text(record.get("ml_raw_drowsy_probability")))
    c12.metric("Live alert probability", safe_percent_text(record.get("ml_alert_probability")))


def render_status(record):
    st.subheader("Current Status")

    if record is None:
        st.info("Click START in the WebRTC camera section and allow camera access.")
        return

    st.write(
        {
            "driver_status": record.get("driver_status", "-"),
            "face_detected": bool(record.get("face_detected", False)),
            "warning_signs": safe_int(record.get("warning_signs")),
            "sample_time": safe_float_text(record.get("time"), digits=1),
            "ml_prediction": record.get("ml_prediction", "-"),
            "ml_live_drowsy_probability": safe_percent_text(record.get("ml_drowsy_probability")),
            "ml_raw_drowsy_probability": safe_percent_text(record.get("ml_raw_drowsy_probability")),
            "ml_live_evidence": safe_percent_text(record.get("ml_live_evidence")),
            "rule_vs_ml": f"{record.get('driver_status', '-')} vs {record.get('ml_prediction', '-')}",
        }
    )


def render_charts(df):
    st.subheader("Live Signals")

    if df.empty:
        st.info("No telemetry points collected yet.")
        return

    left, right = st.columns(2)

    with left:
        st.altair_chart(
            make_line_chart(
                df,
                ["ear", "ear_threshold"],
                "Eye Aspect Ratio vs Adaptive Threshold",
                "EAR",
            ),
            use_container_width=True,
        )
        st.altair_chart(
            make_line_chart(
                df,
                ["score"],
                "Drowsiness Risk Score",
                "Score",
                rules=[
                    {"value": 30, "label": "Visual warning", "color": "#f59e0b"},
                    {"value": 60, "label": "Audio alert", "color": "#f97316"},
                    {"value": 80, "label": "Critical", "color": "#ef4444"},
                ],
            ),
            use_container_width=True,
        )

    with right:
        st.altair_chart(
            make_line_chart(
                df,
                ["mar"],
                "Mouth Aspect Ratio vs Yawn Threshold",
                "MAR",
                rules=[{"value": YAWN_MAR_THRESHOLD, "label": "Yawn threshold"}],
            ),
            use_container_width=True,
        )
        st.altair_chart(
            make_line_chart(
                df,
                ["pitch", "yaw", "roll"],
                "Head Pose Angles",
                "Degrees",
                rules=[
                    {"value": HEAD_DOWN_PITCH_THRESHOLD, "label": "Head down"},
                    {"value": HEAD_UP_PITCH_THRESHOLD, "label": "Head up"},
                    {"value": SIDE_LOOK_YAW_THRESHOLD, "label": "Yaw limit"},
                    {"value": -SIDE_LOOK_YAW_THRESHOLD, "label": "Yaw limit"},
                    {"value": HEAD_TILT_ROLL_THRESHOLD, "label": "Roll limit"},
                    {"value": -HEAD_TILT_ROLL_THRESHOLD, "label": "Roll limit"},
                ],
            ),
            use_container_width=True,
        )
        st.altair_chart(
            make_line_chart(
                df,
                ["ml_drowsy_probability", "ml_raw_drowsy_probability"],
                "ML Drowsy Probability: Live Calibrated vs Raw Model",
                "Probability",
                rules=[
                    {
                        "value": ML_DROWSY_PROBABILITY_THRESHOLD,
                        "label": "Live drowsy threshold",
                        "color": "#38bdf8",
                    }
                ],
            ),
            use_container_width=True,
        )


def render_recent_telemetry(df):
    st.subheader("Recent Telemetry")

    if df.empty:
        st.info("No telemetry yet.")
        return

    visible_columns = [
        "time",
        "ear",
        "ear_threshold",
        "mar",
        "pitch",
        "yaw",
        "roll",
        "score",
        "rule_based_score",
        "rule_based_status",
        "rule_based_reasons",
        "ml_prediction",
        "final_drowsiness_probability",
        "ml_drowsy_probability",
        "ml_raw_drowsy_probability",
        "ml_alert_probability",
        "ml_live_evidence",
        "ml_drowsy_threshold",
        "temporal_10s_perclos_percent",
        "temporal_30s_perclos_percent",
        "temporal_60s_perclos_percent",
        "temporal_60s_perclos_duration",
        "temporal_60s_closed_eye_ratio",
        "temporal_60s_heavy_eye_ratio",
        "temporal_60s_head_down_duration",
        "temporal_60s_side_look_duration",
        "temporal_60s_head_abnormal_duration",
        "temporal_60s_longest_head_down_duration",
        "temporal_60s_longest_side_look_duration",
        "temporal_60s_head_pose_instability",
        "head_down_duration",
        "side_look_duration",
        "head_pose_instability",
        "temporal_60s_blink_count",
        "temporal_60s_blink_rate_per_minute",
        "temporal_60s_avg_blink_duration",
        "temporal_60s_longest_eye_closure",
        "temporal_60s_slow_blink_count",
        "temporal_60s_microsleep_event_count",
        "temporal_60s_microsleep_active",
        "temporal_60s_microsleep_recent",
        "temporal_60s_seconds_since_last_microsleep",
        "temporal_60s_microsleep_detected",
        "yawn_count_60s",
        "yawn_count_120s",
        "avg_yawn_duration",
        "avg_yawn_duration_60s",
        "avg_yawn_duration_120s",
        "max_yawn_duration",
        "seconds_since_last_yawn",
        "fatigue_signs",
        "driver_status",
        "eye_status",
        "mouth_status",
        "head_status",
    ]
    visible_columns = [column for column in visible_columns if column in df.columns]
    st.dataframe(df[visible_columns].tail(15), use_container_width=True, hide_index=True)


def render_dashboard():
    st.set_page_config(
        page_title="SafeDrive AI Dashboard",
        page_icon="SD",
        layout="wide",
    )
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem; }
            [data-testid="stMetric"] {
                background: #101827;
                border: 1px solid #263247;
                border-radius: 8px;
                padding: 14px 16px;
            }
            [data-testid="stMetricLabel"] { color: #a7b4c8; }
            [data-testid="stMetricValue"] { color: #f8fafc; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("SafeDrive AI Live Dashboard")
    st.caption(
        f"The browser camera stream is processed with streamlit-webrtc. "
        f"The OpenCV app still uses CAMERA_INDEX={CAMERA_INDEX} if you run main.py separately."
    )

    render_webrtc_camera()

    @st.fragment(run_every=TELEMETRY_SAMPLE_SECONDS)
    def telemetry_fragment():
        telemetry = read_telemetry()
        record = telemetry.get("latest")
        df = pd.DataFrame(telemetry.get("records", []))

        render_metric_cards(record)
        render_status(record)
        render_charts(df)
        render_recent_telemetry(df)

    telemetry_fragment()


if __name__ == "__main__":
    render_dashboard()
