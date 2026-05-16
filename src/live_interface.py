from collections import deque
import time

import cv2
import mediapipe as mp
import numpy as np

from alert_system import (
    calculate_rule_based_score,
    get_alert_level,
    get_rule_based_driver_status,
    play_alert_sound,
)
from config import (
    ADAPTIVE_EAR_WINDOW,
    CAMERA_INDEX,
    EYE_CLOSED_EAR_THRESHOLD,
    EAR_SMOOTHING_WINDOW,
    MODEL_PATH,
    TELEMETRY_SAMPLE_SECONDS,
    YAWN_MAR_THRESHOLD,
)
from face_landmarks import (
    ABNORMAL_HEAD_STATUSES,
    calculate_effective_ear,
    create_face_landmarker,
    get_smoothed_ear,
    open_camera,
    print_camera_resolution,
    should_update_adaptive_threshold,
    update_adaptive_ear_threshold,
)
from facial_features import calculate_mar, get_eye_status, get_mouth_status
from fatigue_signs import (
    FREQUENT_BLINKING,
    FREQUENT_YAWNING,
    MICROSLEEP,
    PROLONGED_SIDE_LOOK,
    SLOW_BLINKING,
    HEAD_DOWN,
    UNSTABLE_HEAD_POSE,
    classify_instant_fatigue_signs,
    classify_temporal_fatigue_signs,
    format_sign_labels,
    merge_fatigue_signs,
)
from head_pose import estimate_head_pose, get_head_status
from landmark_indexes import MOUTH_MAR_POINTS
from ml_predictor import load_ml_model, predict_drowsiness
from telemetry_store import reset_telemetry, update_telemetry
from temporal_analysis import TemporalFatigueAnalyzer


WINDOW_NAME = "SafeDrive AI"
DASHBOARD_WIDTH = 1536
DASHBOARD_HEIGHT = 960
APP_NAME = "SafeDrive AI"

COLOR_BG_TOP = (31, 19, 8)
COLOR_BG_BOTTOM = (18, 10, 4)
COLOR_PANEL = (42, 30, 17)
COLOR_PANEL_DARK = (30, 20, 10)
COLOR_PANEL_BORDER = (78, 55, 34)
COLOR_GREEN = (82, 220, 112)
COLOR_YELLOW = (28, 184, 255)
COLOR_ORANGE = (38, 142, 255)
COLOR_RED = (54, 72, 245)
COLOR_TEXT = (238, 244, 252)
COLOR_MUTED = (155, 166, 182)
COLOR_DIM = (95, 108, 128)
COLOR_BACKGROUND = (80, 50, 45)
COLOR_WHITE = (255, 255, 255)

MAIN_W = 1000
VIDEO_H = 562
MAIN_X = (DASHBOARD_WIDTH - MAIN_W) // 2
MAIN_Y = 150
MAIN_H = VIDEO_H
SCORE_Y = MAIN_Y + VIDEO_H + 70

SIDEBAR_X = 21
SIDEBAR_Y = 130
SIDEBAR_W = 142
SIDEBAR_H = 830

RIGHT_X = 1212
RIGHT_W = 286
CARD_GAP = 16


def get_ml_alert_level(ml_probability):
    if ml_probability is None:
        return 0

    if ml_probability >= 0.80:
        return 3

    if ml_probability >= 0.65:
        return 2

    if ml_probability >= 0.40:
        return 1

    return 0


def get_temporal_alert_level(fatigue_sign_ids):
    if MICROSLEEP in fatigue_sign_ids:
        return 3

    if SLOW_BLINKING in fatigue_sign_ids:
        return 1

    if FREQUENT_YAWNING in fatigue_sign_ids:
        return 2

    if HEAD_DOWN in fatigue_sign_ids or PROLONGED_SIDE_LOOK in fatigue_sign_ids:
        return 1

    if UNSTABLE_HEAD_POSE in fatigue_sign_ids:
        return 1

    if FREQUENT_BLINKING in fatigue_sign_ids:
        return 1

    return 0


def get_risk_color(ml_probability):
    if ml_probability is None:
        return COLOR_MUTED

    if ml_probability >= 0.80:
        return COLOR_RED

    if ml_probability >= 0.65:
        return COLOR_ORANGE

    if ml_probability >= 0.40:
        return COLOR_YELLOW

    return COLOR_GREEN


def get_risk_label(ml_probability):
    if ml_probability is None:
        return "UNKNOWN"

    if ml_probability >= 0.80:
        return "CRITICAL"

    if ml_probability >= 0.65:
        return "HIGH"

    if ml_probability >= 0.40:
        return "MODERATE"

    return "LOW"


def get_status_text(ml_probability, face_detected):
    if not face_detected:
        return "Face not detected", "Adjust camera position."

    if ml_probability is None:
        return "Model unavailable", "Check the ML model."

    if ml_probability >= 0.80:
        return "Critical alert", "Stop safely."

    if ml_probability >= 0.65:
        return "Fatigue detected", "Stay focused on the road."

    if ml_probability >= 0.40:
        return "Increased attention", "Consider taking a break."

    return "Driver alert", "Monitoring active."


def get_advice_text(ml_probability):
    if ml_probability is None:
        return "Keep your face visible for an accurate estimate."

    if ml_probability >= 0.80:
        return "Stop as soon as possible and take a break."

    if ml_probability >= 0.50:
        return "Rest at the next safe stop."

    if ml_probability >= 0.30:
        return "Keep looking forward and maintain a stable posture."

    return "Continue monitoring under normal conditions."


def blend_colors(start_color, end_color, ratio):
    return tuple(
        int(start + (end - start) * ratio)
        for start, end in zip(start_color, end_color)
    )


def draw_vertical_gradient(frame, top_color, bottom_color):
    height = frame.shape[0]

    for offset in range(height):
        ratio = offset / max(1, height - 1)
        color = blend_colors(top_color, bottom_color, ratio)
        cv2.line(frame, (0, offset), (frame.shape[1], offset), color, 1)


def draw_rounded_rect(frame, top_left, bottom_right, color, radius=16, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = min(radius, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))

    if thickness == -1:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
        return

    cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_text(frame, text, position, color, scale=1.0, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_wrapped_text(frame, text, x, y, max_width, color, scale=0.45, thickness=1, line_gap=26):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        candidate = word if not current_line else f"{current_line} {word}"
        candidate_size, _ = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )
        candidate_width = candidate_size[0]

        if candidate_width <= max_width or not current_line:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    for index, line in enumerate(lines[:3]):
        draw_text(frame, line, (x, y + index * line_gap), color, scale, thickness)


def text_size(text, scale=1.0, thickness=2):
    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return size[0], size[1], baseline


def draw_centered_text(frame, text, center_x, y, color, scale=1.0, thickness=2):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(center_x - text_size[0] / 2)
    draw_text(frame, text, (x, y), color, scale, thickness)


def draw_card(canvas, x, y, width, height, radius=18, color=COLOR_PANEL):
    draw_rounded_rect(canvas, (x, y), (x + width, y + height), color, radius)
    draw_rounded_rect(
        canvas,
        (x, y),
        (x + width, y + height),
        COLOR_PANEL_BORDER,
        radius,
        thickness=1,
    )


def paste_rounded_image(canvas, image, x, y, width, height, radius=14):
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask = np.zeros((height, width), dtype=np.uint8)
    draw_rounded_rect(mask, (0, 0), (width - 1, height - 1), 255, radius)
    roi = canvas[y:y + height, x:x + width]
    mask_3d = cv2.merge([mask, mask, mask]) / 255.0
    blended = (resized_image * mask_3d + roi * (1.0 - mask_3d)).astype(np.uint8)
    canvas[y:y + height, x:x + width] = blended


def resize_to_cover(image, target_width, target_height):
    source_height, source_width, _ = image.shape
    scale = max(target_width / source_width, target_height / source_height)
    resized_width = int(source_width * scale)
    resized_height = int(source_height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    crop_x = max(0, (resized_width - target_width) // 2)
    crop_y = max(0, (resized_height - target_height) // 2)
    cropped = resized[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
    return cropped, scale, crop_x, crop_y


def draw_camera_overlay(canvas, landmarks, source_shape, scale, crop_x, crop_y, offset_x, offset_y):
    if not landmarks:
        return

    source_height, source_width = source_shape[:2]
    points = []

    for landmark in landmarks:
        x = int(landmark.x * source_width * scale - crop_x + offset_x)
        y = int(landmark.y * source_height * scale - crop_y + offset_y)
        points.append((x, y))

    visible_points = [
        point for point in points
        if offset_x <= point[0] <= offset_x + MAIN_W and offset_y <= point[1] <= offset_y + VIDEO_H
    ]
    if not visible_points:
        return

    min_x = max(offset_x + 20, min(point[0] for point in visible_points))
    max_x = min(offset_x + MAIN_W - 20, max(point[0] for point in visible_points))
    min_y = max(offset_y + 20, min(point[1] for point in visible_points))
    max_y = min(offset_y + VIDEO_H - 20, max(point[1] for point in visible_points))

    bracket = 26
    thickness = 3
    color = COLOR_GREEN
    corners = [
        ((min_x, min_y), (min_x + bracket, min_y), (min_x, min_y + bracket)),
        ((max_x, min_y), (max_x - bracket, min_y), (max_x, min_y + bracket)),
        ((min_x, max_y), (min_x + bracket, max_y), (min_x, max_y - bracket)),
        ((max_x, max_y), (max_x - bracket, max_y), (max_x, max_y - bracket)),
    ]

    for corner, horizontal, vertical in corners:
        cv2.line(canvas, corner, horizontal, color, thickness)
        cv2.line(canvas, corner, vertical, color, thickness)

    for point in visible_points[::12]:
        cv2.circle(canvas, point, 2, (116, 226, 129), -1)


def draw_logo(canvas):
    draw_text(canvas, APP_NAME, (44, 72), COLOR_WHITE, 0.72, 2)


def draw_sidebar_icon(canvas, icon_name, center, color):
    x, y = center
    if icon_name == "camera":
        cv2.rectangle(canvas, (x - 18, y - 11), (x + 9, y + 11), color, -1)
        pts = np.array([[x + 9, y - 7], [x + 22, y - 14], [x + 22, y + 14], [x + 9, y + 7]])
        cv2.fillConvexPoly(canvas, pts, color)
        cv2.circle(canvas, (x - 5, y), 4, COLOR_PANEL_DARK, -1)
    elif icon_name == "history":
        for index, height in enumerate((12, 25, 35)):
            bar_x = x - 15 + index * 10
            cv2.rectangle(canvas, (bar_x, y + 18 - height), (bar_x + 5, y + 18), color, -1)
    elif icon_name == "settings":
        cv2.circle(canvas, (x, y), 14, color, 3)
        cv2.circle(canvas, (x, y), 5, COLOR_PANEL_DARK, -1)
        for angle in range(0, 360, 60):
            radians = np.deg2rad(angle)
            x1 = int(x + np.cos(radians) * 18)
            y1 = int(y + np.sin(radians) * 18)
            x2 = int(x + np.cos(radians) * 23)
            y2 = int(y + np.sin(radians) * 23)
            cv2.line(canvas, (x1, y1), (x2, y2), color, 3)
    elif icon_name == "info":
        cv2.circle(canvas, (x, y), 14, color, 2)
        draw_centered_text(canvas, "i", x, y + 8, color, 0.8, 2)


def draw_sidebar(canvas):
    draw_card(canvas, SIDEBAR_X, SIDEBAR_Y, SIDEBAR_W, SIDEBAR_H, radius=18, color=COLOR_PANEL_DARK)
    active_x = SIDEBAR_X + 14
    active_y = SIDEBAR_Y + 24
    active_w = SIDEBAR_W - 28
    active_h = 164
    draw_rounded_rect(
        canvas,
        (active_x, active_y),
        (active_x + active_w, active_y + active_h),
        (17, 63, 48),
        radius=15,
    )
    draw_sidebar_icon(canvas, "camera", (SIDEBAR_X + SIDEBAR_W // 2, active_y + 63), COLOR_GREEN)
    draw_centered_text(canvas, "Live Monitor", SIDEBAR_X + SIDEBAR_W // 2, active_y + 116, COLOR_GREEN, 0.47, 1)

    items = [
        ("history", "History", active_y + 295),
        ("settings", "Settings", active_y + 475),
        ("info", "About", active_y + 705),
    ]
    for icon_name, label, y in items:
        draw_sidebar_icon(canvas, icon_name, (SIDEBAR_X + SIDEBAR_W // 2, y), COLOR_MUTED)
        draw_centered_text(canvas, label, SIDEBAR_X + SIDEBAR_W // 2, y + 50, COLOR_MUTED, 0.47, 1)


def draw_live_badge(canvas):
    x = MAIN_X + 30
    y = MAIN_Y + 26
    width = 104
    height = 46
    draw_rounded_rect(canvas, (x, y), (x + width, y + height), (18, 27, 38), radius=14)
    draw_rounded_rect(canvas, (x, y), (x + width, y + height), (103, 120, 140), radius=14, thickness=1)
    cv2.circle(canvas, (x + 28, y + 23), 7, COLOR_GREEN, -1)
    draw_text(canvas, "LIVE", (x + 47, y + 30), COLOR_TEXT, 0.58, 2)


def draw_main_camera(canvas, frame, face_landmarks):
    camera_image, scale, crop_x, crop_y = resize_to_cover(frame, MAIN_W, VIDEO_H)
    paste_rounded_image(canvas, camera_image, MAIN_X, MAIN_Y, MAIN_W, VIDEO_H, radius=12)


def draw_ml_score(canvas, ml_probability):
    percent = "--%" if ml_probability is None else f"{int(round(ml_probability * 100))}%"
    draw_centered_text(
        canvas,
        f"ML fatigue score: {percent}",
        DASHBOARD_WIDTH // 2,
        SCORE_Y,
        COLOR_WHITE,
        0.95,
        2,
    )


def draw_waveform(canvas, x, y, width, height, color, phase=0.0):
    points = []
    for index in range(width):
        ratio = index / max(1, width - 1)
        value = (
            np.sin(ratio * np.pi * 4 + phase) * 0.55
            + np.sin(ratio * np.pi * 11 + phase * 0.7) * 0.25
        )
        px = x + index
        py = int(y + height / 2 - value * height * 0.42)
        points.append((px, py))

    for point_a, point_b in zip(points, points[1:]):
        if point_a[0] % 5 < 3:
            cv2.line(canvas, point_a, point_b, color, 1)


def draw_gauge(canvas, center, radius, probability, color):
    cv2.circle(canvas, center, radius, (31, 42, 58), 2)
    cv2.circle(canvas, center, radius - 18, (18, 28, 43), 1)

    if probability is not None:
        end_angle = -90 + int(300 * min(1.0, max(0.0, probability)))
        cv2.ellipse(canvas, center, (radius - 8, radius - 8), 0, -210, 90, (42, 54, 72), 8)
        cv2.ellipse(canvas, center, (radius - 8, radius - 8), 0, -210, end_angle, color, 8)

    percent = "--" if probability is None else str(int(round(probability * 100)))
    draw_centered_text(canvas, percent, center[0], center[1] + 10, color, 1.55, 4)
    draw_centered_text(canvas, "FINAL RISK", center[0], center[1] + 45, color, 0.44, 1)


def draw_main_bottom(canvas, ml_probability):
    bottom_y = MAIN_Y + VIDEO_H
    color = get_risk_color(ml_probability)
    label = get_risk_label(ml_probability)

    cv2.line(canvas, (MAIN_X, bottom_y), (MAIN_X + MAIN_W, bottom_y), (26, 38, 54), 1)

    left_x = MAIN_X + 40
    draw_text(canvas, "Fatigue level", (left_x, bottom_y + 58), COLOR_TEXT, 0.52, 1)
    draw_text(canvas, label, (left_x, bottom_y + 92), color, 0.74, 2)
    draw_waveform(canvas, left_x + 170, bottom_y + 45, 210, 80, color)

    gauge_center = (MAIN_X + MAIN_W // 2, bottom_y + 84)
    draw_gauge(canvas, gauge_center, 84, ml_probability, color)


def draw_face_icon(canvas, center, radius, color):
    cv2.circle(canvas, center, radius, color, -1)
    eye_y = center[1] - 10
    cv2.ellipse(canvas, (center[0] - 18, eye_y), (8, 5), 0, 0, 180, COLOR_PANEL_DARK, 3)
    cv2.ellipse(canvas, (center[0] + 18, eye_y), (8, 5), 0, 0, 180, COLOR_PANEL_DARK, 3)
    cv2.ellipse(canvas, (center[0], center[1] + 22), (16, 10), 0, 180, 360, COLOR_PANEL_DARK, 5)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02d}"


def draw_clock_icon(canvas, center, color):
    cv2.circle(canvas, center, 24, color, 3)
    cv2.line(canvas, center, (center[0], center[1] - 14), color, 3)
    cv2.line(canvas, center, (center[0] + 12, center[1] + 8), color, 3)


def draw_break_icon(canvas, center, color):
    x, y = center
    cv2.rectangle(canvas, (x - 18, y - 8), (x + 14, y + 24), color, 3)
    cv2.ellipse(canvas, (x + 18, y + 8), (10, 12), 270, -70, 70, color, 3)
    cv2.line(canvas, (x - 22, y + 30), (x + 20, y + 30), color, 3)
    for offset in (-12, 0, 12):
        cv2.line(canvas, (x + offset, y - 18), (x + offset + 4, y - 30), color, 2)


def draw_session_chart(canvas, x, y, width, height, history, color):
    cv2.line(canvas, (x, y + height - 18), (x + width, y + height - 18), (25, 36, 52), 1)
    if not history:
        return

    values = list(history)[-32:]
    if len(values) < 2:
        values = values * 2

    points = []
    for index, value in enumerate(values):
        ratio = index / max(1, len(values) - 1)
        px = x + int(ratio * width)
        py = y + int((1.0 - min(1.0, max(0.0, value))) * (height - 28))
        points.append((px, py))

    for point_a, point_b in zip(points, points[1:]):
        cv2.line(canvas, point_a, point_b, color, 2)
    cv2.circle(canvas, points[-1], 5, color, -1)


def draw_right_cards(canvas, ml_probability, face_detected, elapsed_seconds, risk_history):
    status, subtitle = get_status_text(ml_probability, face_detected)
    color = get_risk_color(ml_probability)
    card_height = 230
    y = MAIN_Y

    draw_card(canvas, RIGHT_X, y, RIGHT_W, card_height, radius=18)
    draw_text(canvas, "Current status", (RIGHT_X + 32, y + 46), COLOR_TEXT, 0.55, 1)
    draw_centered_text(canvas, status, RIGHT_X + RIGHT_W // 2, y + 118, color, 0.6, 2)
    draw_centered_text(canvas, subtitle, RIGHT_X + RIGHT_W // 2, y + 158, COLOR_MUTED, 0.43, 1)


def label_for_chart(ml_probability):
    return get_risk_label(ml_probability).title()


def compose_interface_frame(
    frame,
    ml_result,
    face_detected,
    face_landmarks=None,
    elapsed_seconds=0.0,
    risk_history=None,
):
    ml_probability = ml_result.get(
        "final_drowsiness_probability",
        ml_result.get("ml_drowsy_probability"),
    )
    canvas = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    canvas[:] = COLOR_BACKGROUND
    draw_logo(canvas)
    draw_main_camera(canvas, frame, face_landmarks)
    draw_ml_score(canvas, ml_probability)
    return canvas


def unavailable_ml_result(ml_model):
    return {
        "ml_model_name": ml_model["model_name"] if ml_model else None,
        "ml_prediction": "Unavailable",
        "ml_confidence": None,
        "ml_drowsy_probability": None,
        "final_drowsiness_probability": None,
        "ml_calibrated_drowsy_probability": None,
        "ml_raw_drowsy_probability": None,
        "ml_alert_probability": None,
        "ml_raw_alert_probability": None,
        "ml_live_evidence": None,
        "ml_drowsy_threshold": None,
    }


def analyze_face(
    frame,
    face_landmarks,
    closed_start_time,
    heavy_start_time,
    last_sound_time,
    ear_history,
    adaptive_ear_history,
    ear_threshold,
    session_start_time,
    last_telemetry_sample_time,
    ml_model,
    temporal_analyzer=None,
):
    raw_ear, left_ear, right_ear = calculate_effective_ear(frame, face_landmarks)
    average_ear = get_smoothed_ear(raw_ear, ear_history)
    eye_status, _, closed_start_time, heavy_start_time = get_eye_status(
        average_ear,
        closed_start_time,
        ear_threshold,
        heavy_start_time,
    )
    current_time = time.time()
    current_eye_closed_seconds = (
        current_time - closed_start_time
        if closed_start_time is not None and eye_status in {"Eyes closed", "Drowsiness warning"}
        else 0.0
    )
    current_heavy_eye_seconds = (
        current_time - heavy_start_time
        if heavy_start_time is not None and eye_status == "Eyes heavy"
        else 0.0
    )

    mar = calculate_mar(frame, face_landmarks, MOUTH_MAR_POINTS)
    mouth_status, _ = get_mouth_status(mar)

    head_pose = estimate_head_pose(frame, face_landmarks)
    head_status, _ = get_head_status(head_pose)
    instant_fatigue_sign_ids = classify_instant_fatigue_signs(
        eye_status,
        mouth_status,
        head_status,
    )

    open_eye_reference = None
    if should_update_adaptive_threshold(
        eye_status,
        mouth_status,
        head_status,
        average_ear,
        ear_threshold,
    ):
        ear_threshold, open_eye_reference = update_adaptive_ear_threshold(
            raw_ear,
            adaptive_ear_history,
            ear_threshold,
        )

    pitch, yaw, roll = head_pose if head_pose is not None else (None, None, None)

    elapsed = time.time() - session_start_time
    should_sample = elapsed - last_telemetry_sample_time >= TELEMETRY_SAMPLE_SECONDS
    if should_sample:
        last_telemetry_sample_time = elapsed

    temporal_features = {}
    if temporal_analyzer is not None:
        temporal_features = temporal_analyzer.add_sample(
            timestamp=elapsed,
            face_detected=True,
            ear=average_ear,
            raw_ear=raw_ear,
            left_ear=left_ear,
            right_ear=right_ear,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            eye_status=eye_status,
            mouth_status=mouth_status,
            head_status=head_status,
        )

    temporal_fatigue_sign_ids = classify_temporal_fatigue_signs(temporal_features)
    fatigue_sign_ids = merge_fatigue_signs(
        instant_fatigue_sign_ids,
        temporal_fatigue_sign_ids,
    )
    scoring_features = {
        **temporal_features,
        "current_eye_closed_seconds": current_eye_closed_seconds,
        "current_heavy_eye_seconds": current_heavy_eye_seconds,
    }
    drowsiness_score, rule_warning_signs = calculate_rule_based_score(
        eye_status,
        mouth_status,
        head_status,
        scoring_features,
        fatigue_sign_ids,
    )
    driver_status, _, _ = get_rule_based_driver_status(
        eye_status,
        mouth_status,
        head_status,
        drowsiness_score,
        rule_warning_signs,
    )
    _, _, rule_alert_level = get_alert_level(drowsiness_score)

    ml_result = predict_drowsiness(
        ml_model,
        {
            "ear": average_ear,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "score": drowsiness_score,
            "alert_level": rule_alert_level,
            "current_eye_closed_seconds": current_eye_closed_seconds,
            "current_heavy_eye_seconds": current_heavy_eye_seconds,
            "eye_closed_signal": 1.0 if eye_status != "Eyes open" else 0.0,
            "yawn_signal": 1.0 if mouth_status == "Yawning detected" else 0.0,
            "head_abnormal_signal": (
                1.0
                if any(
                    sign_id in fatigue_sign_ids
                    for sign_id in (HEAD_DOWN, PROLONGED_SIDE_LOOK, UNSTABLE_HEAD_POSE)
                )
                else 0.0
            ),
            **temporal_features,
        },
        {
            "ear_threshold": ear_threshold,
            "mar_threshold": YAWN_MAR_THRESHOLD,
            "eye_status": eye_status,
            "mouth_status": mouth_status,
            "head_status": head_status,
            "driver_status": driver_status,
            "score": drowsiness_score,
            "current_eye_closed_seconds": current_eye_closed_seconds,
            "current_heavy_eye_seconds": current_heavy_eye_seconds,
        },
    )
    ml_result.update(
        {
            "rule_based_score": drowsiness_score,
            "rule_based_status": driver_status,
            "rule_based_reasons": rule_warning_signs,
        }
    )

    ml_alert_level = get_ml_alert_level(ml_result.get("ml_drowsy_probability"))
    temporal_alert_level = get_temporal_alert_level(fatigue_sign_ids)
    final_alert_level = max(rule_alert_level, ml_alert_level, temporal_alert_level)
    last_sound_time = play_alert_sound(
        final_alert_level,
        last_sound_time,
    )

    update_telemetry(
        {
            "time": elapsed,
            "face_detected": True,
            "ear": average_ear,
            "raw_ear": raw_ear,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "open_eye_reference": open_eye_reference,
            "ear_threshold": ear_threshold,
            "mar": mar,
            "mar_threshold": YAWN_MAR_THRESHOLD,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "score": drowsiness_score,
            "rule_based_score": drowsiness_score,
            "rule_based_status": driver_status,
            "rule_based_reasons": rule_warning_signs,
            "current_eye_closed_seconds": current_eye_closed_seconds,
            "current_heavy_eye_seconds": current_heavy_eye_seconds,
            "alert_level": final_alert_level,
            "warning_signs": len(rule_warning_signs),
            "driver_status": driver_status,
            "eye_status": eye_status,
            "mouth_status": mouth_status,
            "head_status": head_status,
            "fatigue_sign_ids": fatigue_sign_ids,
            "fatigue_signs": format_sign_labels(fatigue_sign_ids),
            **temporal_features,
            **ml_result,
        },
        should_sample,
    )

    return (
        ml_result,
        closed_start_time,
        heavy_start_time,
        last_sound_time,
        ear_threshold,
        last_telemetry_sample_time,
    )


def update_no_face_telemetry(
    elapsed,
    ear_threshold,
    ml_model,
    last_telemetry_sample_time,
    temporal_analyzer=None,
):
    should_sample = elapsed - last_telemetry_sample_time >= TELEMETRY_SAMPLE_SECONDS
    if should_sample:
        last_telemetry_sample_time = elapsed

    temporal_features = {}
    if temporal_analyzer is not None:
        temporal_features = temporal_analyzer.add_sample(
            timestamp=elapsed,
            face_detected=False,
            ear=None,
            raw_ear=None,
            left_ear=None,
            right_ear=None,
            mar=None,
            pitch=None,
            yaw=None,
            roll=None,
            eye_status="Unavailable",
            mouth_status="Unavailable",
            head_status="Unavailable",
        )

    update_telemetry(
        {
            "time": elapsed,
            "face_detected": False,
            "ear": None,
            "raw_ear": None,
            "left_ear": None,
            "right_ear": None,
            "open_eye_reference": None,
            "ear_threshold": ear_threshold,
            "mar": None,
            "mar_threshold": YAWN_MAR_THRESHOLD,
            "pitch": None,
            "yaw": None,
            "roll": None,
            "score": 0,
            "rule_based_score": 0,
            "rule_based_status": "No face detected",
            "rule_based_reasons": [],
            "alert_level": 0,
            "warning_signs": 0,
            "driver_status": "No face detected",
            "eye_status": "Unavailable",
            "mouth_status": "Unavailable",
            "head_status": "Unavailable",
            "fatigue_sign_ids": [],
            "fatigue_signs": [],
            **temporal_features,
            **unavailable_ml_result(ml_model),
        },
        should_sample,
    )

    return last_telemetry_sample_time


def run_live_interface():
    if not MODEL_PATH.exists():
        print(f"Missing model: {MODEL_PATH}")
        return

    cap = open_camera()
    if not cap.isOpened():
        print(f"Camera {CAMERA_INDEX} unavailable.")
        return

    print_camera_resolution(cap)
    landmarker = create_face_landmarker()
    ml_model = load_ml_model()

    print("SafeDrive AI interface started. Press Q or Esc to exit.")
    if ml_model is None:
        print("ML model unavailable.")
    else:
        print(f"ML model loaded: {ml_model['model_name']}")

    reset_telemetry()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    closed_start_time = None
    heavy_start_time = None
    last_sound_time = 0.0
    ear_history = deque(maxlen=EAR_SMOOTHING_WINDOW)
    adaptive_ear_history = deque(maxlen=ADAPTIVE_EAR_WINDOW)
    ear_threshold = EYE_CLOSED_EAR_THRESHOLD
    session_start_time = time.time()
    last_telemetry_sample_time = 0.0
    ml_result = unavailable_ml_result(ml_model)
    risk_history = deque(maxlen=64)
    temporal_analyzer = TemporalFatigueAnalyzer()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Frame read failed.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame),
                int(time.time() * 1000),
            )

            face_detected = bool(result.face_landmarks)
            face_landmarks = result.face_landmarks[0] if face_detected else None
            if face_detected:
                (
                    ml_result,
                    closed_start_time,
                    heavy_start_time,
                    last_sound_time,
                    ear_threshold,
                    last_telemetry_sample_time,
                ) = analyze_face(
                    frame,
                    face_landmarks,
                    closed_start_time,
                    heavy_start_time,
                    last_sound_time,
                    ear_history,
                    adaptive_ear_history,
                    ear_threshold,
                    session_start_time,
                    last_telemetry_sample_time,
                    ml_model,
                    temporal_analyzer,
                )
            else:
                closed_start_time = None
                heavy_start_time = None
                ml_result = unavailable_ml_result(ml_model)
                elapsed = time.time() - session_start_time
                last_telemetry_sample_time = update_no_face_telemetry(
                    elapsed,
                    ear_threshold,
                    ml_model,
                    last_telemetry_sample_time,
                    temporal_analyzer,
                )

            ml_probability = ml_result.get(
                "final_drowsiness_probability",
                ml_result.get("ml_drowsy_probability"),
            )
            if ml_probability is not None:
                risk_history.append(ml_probability)

            elapsed = time.time() - session_start_time
            interface_frame = compose_interface_frame(
                frame,
                ml_result,
                face_detected,
                face_landmarks,
                elapsed,
                risk_history,
            )
            cv2.imshow(WINDOW_NAME, interface_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_interface()
