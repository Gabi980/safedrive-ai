from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    HEAD_DOWN_PITCH_THRESHOLD,
    HEAD_TILT_ROLL_THRESHOLD,
    HEAD_UP_PITCH_THRESHOLD,
    SIDE_LOOK_YAW_THRESHOLD,
    TELEMETRY_SAMPLE_SECONDS,
    YAWN_MAR_THRESHOLD,
)
from telemetry_store import read_telemetry  # noqa: E402


def safe_int(value, default=0):
    if value is None or pd.isna(value):
        return default

    return int(value)


def safe_float_text(value, digits=2):
    if value is None or pd.isna(value):
        return "-"

    return f"{float(value):.{digits}f}"


def make_line_chart(df, value_columns, title, y_title, rules=None, height=250):
    if df.empty:
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


def render_metric_cards(record):
    if record is None:
        st.info("Waiting for data from the OpenCV camera window.")
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


def render_status(record):
    st.subheader("Current Status")

    if record is None:
        st.info("Run `main.py` and keep the camera window open.")
        return

    st.write(
        {
            "driver_status": record.get("driver_status", "-"),
            "face_detected": bool(record.get("face_detected", False)),
            "warning_signs": safe_int(record.get("warning_signs")),
            "sample_time": safe_float_text(record.get("time"), digits=1),
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
        "driver_status",
        "eye_status",
        "mouth_status",
        "head_status",
    ]
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

    st.title("SafeDrive AI Telemetry Dashboard")
    st.caption("The camera runs in the OpenCV window. This dashboard only displays live telemetry.")

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
