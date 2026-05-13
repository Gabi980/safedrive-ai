import joblib
import pandas as pd

from config import (
    BLINK_RATE_MIN_OBSERVATION_SECONDS,
    FREQUENT_BLINKS_PER_MINUTE_THRESHOLD,
    FREQUENT_YAWN_COUNT_THRESHOLD,
    HEAD_POSE_INSTABILITY_THRESHOLD,
    LONG_YAWN_SECONDS_THRESHOLD,
    MICROSLEEP_SECONDS_THRESHOLD,
    ML_DROWSY_PROBABILITY_THRESHOLD,
    ML_MODEL_PATH,
    PROLONGED_HEAD_DOWN_SECONDS,
    PROLONGED_SIDE_LOOK_SECONDS,
    SLOW_BLINK_COUNT_THRESHOLD,
    SLOW_BLINK_SECONDS_THRESHOLD,
    YAWN_MAR_THRESHOLD,
)


ABNORMAL_HEAD_STATUSES = {"Head down", "Looking sideways", "Head tilted"}


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def number_or_none(value):
    if value is None:
        return None

    return float(value)


def unavailable_result(model_name=None):
    return {
        "ml_model_name": model_name,
        "ml_prediction": "Unavailable",
        "ml_confidence": None,
        "ml_drowsy_probability": None,
        "final_drowsiness_probability": None,
        "ml_calibrated_drowsy_probability": None,
        "ml_raw_drowsy_probability": None,
        "ml_live_evidence": None,
        "ml_alert_probability": None,
        "ml_raw_alert_probability": None,
        "ml_drowsy_threshold": ML_DROWSY_PROBABILITY_THRESHOLD,
    }


def load_ml_model():
    if not ML_MODEL_PATH.exists():
        return None

    return joblib.load(ML_MODEL_PATH)


def calculate_live_evidence(feature_values, live_context):
    if not live_context:
        return None

    ear = feature_values.get("ear")
    mar = feature_values.get("mar")
    ear_threshold = live_context.get("ear_threshold")
    mar_threshold = live_context.get("mar_threshold", YAWN_MAR_THRESHOLD)
    eye_status = live_context.get("eye_status")
    mouth_status = live_context.get("mouth_status")
    head_status = live_context.get("head_status")
    driver_status = live_context.get("driver_status")
    score = live_context.get("score", 0) or 0

    eye_evidence = 0.0
    if ear is not None and ear_threshold:
        eye_drop = (ear_threshold - ear) / (ear_threshold * 0.35)
        eye_evidence = clamp(eye_drop)

    if eye_status == "Eyes heavy":
        eye_evidence = max(eye_evidence, 0.20)
    elif eye_status == "Eyes closed":
        eye_evidence = 0.35
    elif eye_status == "Drowsiness warning":
        eye_evidence = max(eye_evidence, 0.92)

    face_visible_duration = number_or_none(
        feature_values.get("temporal_60s_face_visible_duration")
    )
    blink_rate = number_or_none(feature_values.get("temporal_60s_blink_rate_per_minute"))
    avg_blink_duration = number_or_none(
        feature_values.get("temporal_60s_avg_blink_duration")
    )
    longest_eye_closure = number_or_none(
        feature_values.get("temporal_60s_longest_eye_closure")
    )
    slow_blink_count = number_or_none(feature_values.get("temporal_60s_slow_blink_count"))
    microsleep_active = bool(feature_values.get("temporal_60s_microsleep_active"))
    microsleep_recent = bool(feature_values.get("temporal_60s_microsleep_recent"))

    if microsleep_active or microsleep_recent:
        eye_evidence = max(eye_evidence, 0.95)
    elif (
        (
            slow_blink_count is not None
            and slow_blink_count >= SLOW_BLINK_COUNT_THRESHOLD
        )
        or (
            avg_blink_duration is not None
            and avg_blink_duration >= SLOW_BLINK_SECONDS_THRESHOLD
            and slow_blink_count is not None
            and slow_blink_count >= SLOW_BLINK_COUNT_THRESHOLD
        )
        or (
            longest_eye_closure is not None
            and longest_eye_closure >= SLOW_BLINK_SECONDS_THRESHOLD
            and slow_blink_count is not None
            and slow_blink_count >= SLOW_BLINK_COUNT_THRESHOLD
        )
    ):
        eye_evidence = max(eye_evidence, 0.45)

    if (
        face_visible_duration is not None
        and face_visible_duration >= BLINK_RATE_MIN_OBSERVATION_SECONDS
        and blink_rate is not None
        and blink_rate >= FREQUENT_BLINKS_PER_MINUTE_THRESHOLD
    ):
        eye_evidence = max(eye_evidence, 0.30)

    mouth_evidence = 0.0
    if mar is not None and mar_threshold:
        mouth_start = mar_threshold * 0.75
        mouth_range = mar_threshold - mouth_start
        mouth_evidence = clamp((mar - mouth_start) / mouth_range)

    if mouth_status == "Yawning detected":
        mouth_evidence = max(mouth_evidence, 0.55)
    else:
        mouth_evidence = min(mouth_evidence, 0.25)

    yawn_count_60s = number_or_none(feature_values.get("yawn_count_60s"))
    yawn_count_120s = number_or_none(feature_values.get("yawn_count_120s"))
    avg_yawn_duration = number_or_none(feature_values.get("avg_yawn_duration"))
    seconds_since_last_yawn = number_or_none(
        feature_values.get("seconds_since_last_yawn")
    )

    if yawn_count_120s is not None and yawn_count_120s >= FREQUENT_YAWN_COUNT_THRESHOLD:
        mouth_evidence = max(mouth_evidence, 0.72)
    elif yawn_count_60s is not None and yawn_count_60s >= 2:
        mouth_evidence = max(mouth_evidence, 0.58)

    if avg_yawn_duration is not None and avg_yawn_duration >= LONG_YAWN_SECONDS_THRESHOLD:
        mouth_evidence = max(mouth_evidence, 0.35)

    if (
        seconds_since_last_yawn is not None
        and seconds_since_last_yawn <= 20.0
        and (
            (yawn_count_60s is not None and yawn_count_60s > 0)
            or (yawn_count_120s is not None and yawn_count_120s > 0)
        )
    ):
        mouth_evidence = max(mouth_evidence, 0.20)

    head_down_duration = number_or_none(
        feature_values.get("temporal_30s_head_down_duration")
    )
    side_look_duration = number_or_none(
        feature_values.get("temporal_30s_side_look_duration")
    )
    head_pose_instability = number_or_none(
        feature_values.get("temporal_30s_head_pose_instability")
    )

    head_evidence = 0.0
    if (
        head_down_duration is not None
        and head_down_duration >= PROLONGED_HEAD_DOWN_SECONDS
    ):
        head_evidence = max(head_evidence, 0.55)

    if (
        side_look_duration is not None
        and side_look_duration >= PROLONGED_SIDE_LOOK_SECONDS
    ):
        head_evidence = max(head_evidence, 0.45)

    if (
        head_pose_instability is not None
        and head_pose_instability >= HEAD_POSE_INSTABILITY_THRESHOLD
    ):
        head_evidence = max(head_evidence, 0.35)

    if head_evidence == 0.0 and head_status in ABNORMAL_HEAD_STATUSES:
        head_evidence = 0.10

    status_evidence = 0.0
    if driver_status == "Driver drowsy":
        status_evidence = 0.90
    elif isinstance(driver_status, str) and driver_status.startswith("Warning:"):
        status_evidence = 0.35

    score_evidence = clamp(score / 100.0)

    return max(
        eye_evidence,
        mouth_evidence,
        head_evidence,
        status_evidence,
        score_evidence,
    )


def calibrate_live_probability(raw_drowsy_probability, live_evidence):
    if raw_drowsy_probability is None or live_evidence is None:
        return raw_drowsy_probability

    if live_evidence < 0.15:
        return min(raw_drowsy_probability, 0.25)

    if live_evidence < 0.40:
        calibrated = (raw_drowsy_probability * 0.55) + (live_evidence * 0.45)
        return min(calibrated, 0.45)

    if live_evidence < 0.75:
        calibrated = (raw_drowsy_probability * 0.45) + (live_evidence * 0.55)
        return min(calibrated, 0.70)

    calibrated = (raw_drowsy_probability * 0.35) + (live_evidence * 0.65)
    return clamp(calibrated)


def predict_drowsiness(model_bundle, feature_values, live_context=None):
    if model_bundle is None:
        return unavailable_result()

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    label_names = model_bundle["label_names"]
    model_name = model_bundle["model_name"]
    row = {column: feature_values.get(column) for column in feature_columns}

    if any(value is None for value in row.values()):
        return unavailable_result(model_name)

    features = pd.DataFrame([row], columns=feature_columns)

    probabilities = model.predict_proba(features)[0]
    raw_drowsy_probability = float(probabilities[1]) if len(probabilities) > 1 else None
    live_evidence = calculate_live_evidence(feature_values, live_context)
    drowsy_probability = calibrate_live_probability(
        raw_drowsy_probability,
        live_evidence,
    )
    prediction_index = 1 if drowsy_probability >= ML_DROWSY_PROBABILITY_THRESHOLD else 0
    alert_probability = 1.0 - drowsy_probability
    raw_alert_probability = (
        1.0 - raw_drowsy_probability if raw_drowsy_probability is not None else None
    )
    confidence = drowsy_probability if prediction_index == 1 else alert_probability

    return {
        "ml_model_name": model_name,
        "ml_prediction": label_names[prediction_index],
        "ml_confidence": confidence,
        "ml_drowsy_probability": drowsy_probability,
        "final_drowsiness_probability": drowsy_probability,
        "ml_calibrated_drowsy_probability": drowsy_probability,
        "ml_raw_drowsy_probability": raw_drowsy_probability,
        "ml_alert_probability": alert_probability,
        "ml_raw_alert_probability": raw_alert_probability,
        "ml_live_evidence": live_evidence,
        "ml_drowsy_threshold": ML_DROWSY_PROBABILITY_THRESHOLD,
    }
