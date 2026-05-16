from collections import deque
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from alert_system import (
    calculate_rule_based_score,
    get_alert_level,
    get_rule_based_driver_status,
    play_alert_sound,
)
from config import (
    ADAPTIVE_EAR_MIN_SAMPLES,
    ADAPTIVE_EAR_PERCENTILE,
    ADAPTIVE_EAR_RATIO,
    ADAPTIVE_EAR_UPDATE_RATE,
    ADAPTIVE_EAR_WINDOW,
    CAMERA_INDEX,
    EYE_CLOSED_EAR_THRESHOLD,
    EAR_SMOOTHING_WINDOW,
    MODEL_PATH,
    HEAVY_EYE_EAR_RATIO,
    PREFERRED_CAMERA_HEIGHT,
    PREFERRED_CAMERA_WIDTH,
    TELEMETRY_SAMPLE_SECONDS,
    YAWN_MAR_THRESHOLD,
)
from facial_features import calculate_ear, calculate_mar, get_eye_status, get_mouth_status
from fatigue_signs import (
    FREQUENT_BLINKING,
    FREQUENT_YAWNING,
    HEAD_DOWN,
    MICROSLEEP,
    PROLONGED_SIDE_LOOK,
    SLOW_BLINKING,
    UNSTABLE_HEAD_POSE,
    classify_instant_fatigue_signs,
    classify_temporal_fatigue_signs,
    format_sign_labels,
    merge_fatigue_signs,
)
from head_pose import estimate_head_pose, get_head_status
from landmark_indexes import (
    HEAD_POSE_POINTS,
    LEFT_EAR_POINTS,
    LEFT_EYE_POINTS,
    MOUTH_MAR_POINTS,
    MOUTH_POINTS,
    RIGHT_EAR_POINTS,
    RIGHT_EYE_POINTS,
)
from ml_predictor import load_ml_model, predict_drowsiness
from telemetry_store import reset_telemetry, update_telemetry
from temporal_analysis import TemporalFatigueAnalyzer


ABNORMAL_HEAD_STATUSES = {"Head down", "Looking sideways", "Head tilted"}


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


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREFERRED_CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREFERRED_CAMERA_HEIGHT)
    return cap


def print_camera_resolution(cap):
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")


def create_face_landmarker():
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def draw_landmark_points(frame, landmarks, point_indexes, color, radius=3):
    height, width, _ = frame.shape

    for index in point_indexes:
        landmark = landmarks[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), radius, color, -1)


def draw_text(frame, text, position, color, scale=0.8, thickness=2):
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


def draw_feature_points(frame, face_landmarks):
    draw_landmark_points(frame, face_landmarks, LEFT_EYE_POINTS, (0, 255, 0))
    draw_landmark_points(frame, face_landmarks, RIGHT_EYE_POINTS, (0, 255, 0))
    draw_landmark_points(frame, face_landmarks, MOUTH_POINTS, (255, 0, 255))
    draw_landmark_points(frame, face_landmarks, HEAD_POSE_POINTS, (0, 255, 255), radius=4)


def draw_metrics(
    frame,
    average_ear,
    eye_status,
    eye_status_color,
    mar,
    mouth_status,
    mouth_status_color,
    head_pose,
    head_status,
    head_status_color,
    driver_status,
    driver_status_color,
    warning_signs,
    drowsiness_score,
    alert_text,
    alert_color,
    ml_result,
):
    draw_text(frame, f"EAR: {average_ear:.2f}", (20, 70), (255, 255, 255))
    draw_text(frame, eye_status, (20, 105), eye_status_color)
    draw_text(frame, f"MAR: {mar:.2f}", (20, 140), (255, 255, 255))
    draw_text(frame, mouth_status, (20, 175), mouth_status_color)

    if head_pose is not None:
        pitch, yaw, roll = head_pose
        draw_text(
            frame,
            f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
            (20, 210),
            (255, 255, 255),
            scale=0.7,
        )

    draw_text(frame, head_status, (20, 245), head_status_color)
    draw_text(frame, driver_status, (20, 290), driver_status_color, scale=1.0, thickness=3)
    draw_text(frame, f"Signs: {len(warning_signs)}", (20, 325), driver_status_color)
    draw_text(frame, f"Score: {drowsiness_score}/100", (20, 370), alert_color, scale=0.9)
    draw_text(frame, alert_text, (20, 410), alert_color, scale=0.9)
    draw_text(
        frame,
        format_ml_overlay(ml_result),
        (20, 450),
        (255, 255, 255),
        scale=0.8,
    )


def format_ml_overlay(ml_result):
    if ml_result["ml_drowsy_probability"] is None:
        return "ML: unavailable"

    live_percent = ml_result["ml_drowsy_probability"] * 100
    return f"ML: {ml_result['ml_prediction']} D {live_percent:.0f}%"


def calculate_eye_ears(frame, face_landmarks):
    left_ear = calculate_ear(frame, face_landmarks, LEFT_EAR_POINTS)
    right_ear = calculate_ear(frame, face_landmarks, RIGHT_EAR_POINTS)
    return left_ear, right_ear


def calculate_average_ear(frame, face_landmarks):
    left_ear, right_ear = calculate_eye_ears(frame, face_landmarks)
    return (left_ear + right_ear) / 2.0


def calculate_effective_ear(frame, face_landmarks):
    left_ear, right_ear = calculate_eye_ears(frame, face_landmarks)
    return max(left_ear, right_ear), left_ear, right_ear


def get_smoothed_ear(raw_ear, ear_history):
    ear_history.append(raw_ear)
    return sum(ear_history) / len(ear_history)


def percentile(values, percentile_value):
    if not values:
        return None

    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * percentile_value)
    return sorted_values[index]


def update_adaptive_ear_threshold(raw_ear, adaptive_ear_history, current_threshold):
    adaptive_ear_history.append(raw_ear)

    if len(adaptive_ear_history) < ADAPTIVE_EAR_MIN_SAMPLES:
        return current_threshold, None

    open_eye_reference = percentile(
        list(adaptive_ear_history),
        ADAPTIVE_EAR_PERCENTILE,
    )
    target_threshold = open_eye_reference * ADAPTIVE_EAR_RATIO
    updated_threshold = (
        current_threshold * (1.0 - ADAPTIVE_EAR_UPDATE_RATE)
        + target_threshold * ADAPTIVE_EAR_UPDATE_RATE
    )

    return updated_threshold, open_eye_reference


def should_update_adaptive_threshold(
    eye_status,
    mouth_status,
    head_status,
    average_ear=None,
    ear_threshold=None,
):
    confidently_open = eye_status == "Eyes open"
    if average_ear is not None and ear_threshold is not None:
        confidently_open = (
            confidently_open
            and average_ear >= ear_threshold * HEAVY_EYE_EAR_RATIO
        )

    return (
        confidently_open
        and mouth_status == "Mouth normal"
        and head_status == "Head normal"
    )


def process_face(
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
    draw_feature_points(frame, face_landmarks)

    raw_ear, left_ear, right_ear = calculate_effective_ear(frame, face_landmarks)
    average_ear = get_smoothed_ear(raw_ear, ear_history)
    eye_status, eye_status_color, closed_start_time, heavy_start_time = get_eye_status(
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
    mouth_status, mouth_status_color = get_mouth_status(mar)

    head_pose = estimate_head_pose(frame, face_landmarks)
    head_status, head_status_color = get_head_status(head_pose)
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
    driver_status, driver_status_color, warning_signs = get_rule_based_driver_status(
        eye_status,
        mouth_status,
        head_status,
        drowsiness_score,
        rule_warning_signs,
    )
    alert_text, alert_color, alert_level = get_alert_level(drowsiness_score)
    final_alert_level = max(alert_level, get_temporal_alert_level(fatigue_sign_ids))
    ml_result = predict_drowsiness(
        ml_model,
        {
            "ear": average_ear,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "score": drowsiness_score,
            "alert_level": alert_level,
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
    last_sound_time = play_alert_sound(final_alert_level, last_sound_time)

    draw_metrics(
        frame,
        average_ear,
        eye_status,
        eye_status_color,
        mar,
        mouth_status,
        mouth_status_color,
        head_pose,
        head_status,
        head_status_color,
        driver_status,
        driver_status_color,
        warning_signs,
        drowsiness_score,
        alert_text,
        alert_color,
        ml_result,
    )
    draw_text(frame, f"EAR threshold: {ear_threshold:.2f}", (20, 485), (255, 255, 255), scale=0.7)
    if open_eye_reference is not None:
        draw_text(
            frame,
            f"Open-eye ref: {open_eye_reference:.2f}",
            (20, 515),
            (255, 255, 255),
            scale=0.7,
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
        closed_start_time,
        heavy_start_time,
        last_sound_time,
        ear_threshold,
        last_telemetry_sample_time,
    )


def main():
    if not MODEL_PATH.exists():
        print(f"Missing model: {MODEL_PATH}")
        return

    cap = open_camera()

    if not cap.isOpened():
        print("Camera unavailable.")
        return

    print_camera_resolution(cap)
    landmarker = create_face_landmarker()
    ml_model = load_ml_model()
    print("SafeDrive AI started. Press Q or Esc to exit.")
    if ml_model is None:
        print("ML model unavailable.")
    else:
        print(f"ML model loaded: {ml_model['model_name']}")
    reset_telemetry()

    closed_start_time = None
    heavy_start_time = None
    last_sound_time = 0.0
    ear_history = deque(maxlen=EAR_SMOOTHING_WINDOW)
    adaptive_ear_history = deque(maxlen=ADAPTIVE_EAR_WINDOW)
    ear_threshold = EYE_CLOSED_EAR_THRESHOLD
    session_start_time = time.time()
    last_telemetry_sample_time = 0.0
    temporal_analyzer = TemporalFatigueAnalyzer()

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

        if result.face_landmarks:
            draw_text(frame, "Face detected", (20, 35), (0, 255, 0))
            face_landmarks = result.face_landmarks[0]

            (
                closed_start_time,
                heavy_start_time,
                last_sound_time,
                ear_threshold,
                last_telemetry_sample_time,
            ) = process_face(
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
            draw_text(frame, "No face detected", (20, 35), (0, 0, 255))
            closed_start_time = None
            heavy_start_time = None
            elapsed = time.time() - session_start_time
            should_sample = elapsed - last_telemetry_sample_time >= TELEMETRY_SAMPLE_SECONDS
            if should_sample:
                last_telemetry_sample_time = elapsed

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
                    **temporal_features,
                },
                should_sample,
            )

        cv2.imshow("SafeDrive AI - Face Mesh", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
