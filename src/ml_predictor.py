import joblib
import pandas as pd

from config import (
    BLINK_RATE_MIN_OBSERVATION_SECONDS,
    DROWSINESS_SECONDS_THRESHOLD,
    EYE_CLOSURE_CRITICAL_SECONDS,
    EYE_CLOSURE_HIGH_RISK_SECONDS,
    FREQUENT_BLINKS_PER_MINUTE_THRESHOLD,
    FREQUENT_YAWN_COUNT_THRESHOLD,
    HEAD_POSE_INSTABILITY_THRESHOLD,
    LONG_YAWN_SECONDS_THRESHOLD,
    MICROSLEEP_SECONDS_THRESHOLD,
    ML_DROWSY_PROBABILITY_THRESHOLD,
    ML_MODEL_PATH,
    NORMAL_BLINK_GRACE_SECONDS,
    NORMAL_BLINK_RISK_CAP,
    PERCLOS_HIGH_PERCENT,
    PROLONGED_HEAD_DOWN_SECONDS,
    PROLONGED_SIDE_LOOK_SECONDS,
    SHORT_EYE_CLOSURE_RISK_CAP,
    SLOW_BLINK_COUNT_THRESHOLD,
    SLOW_BLINK_SECONDS_THRESHOLD,
    TEMPORAL_SCORING_MIN_SECONDS,
    YAWN_MAR_THRESHOLD,
)


ABNORMAL_HEAD_STATUSES = {"Head down", "Looking sideways", "Head tilted"}


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def number_or_none(value):
    if value is None:
        return None

    return float(value)


def first_number_or_none(*values):
    for value in values:
        if value is not None:
            return float(value)

    return None


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
    current_eye_closed_seconds = number_or_none(
        feature_values.get(
            "current_eye_closed_seconds",
            live_context.get("current_eye_closed_seconds"),
        )
    )
    current_heavy_eye_seconds = number_or_none(
        feature_values.get(
            "current_heavy_eye_seconds",
            live_context.get("current_heavy_eye_seconds"),
        )
    )
    current_eye_closed_seconds = current_eye_closed_seconds or 0.0
    current_heavy_eye_seconds = current_heavy_eye_seconds or 0.0

    eye_evidence = 0.0
    if ear is not None and ear_threshold:
        eye_drop = (ear_threshold - ear) / (ear_threshold * 0.35)
        eye_evidence = clamp(eye_drop)

    if eye_status == "Eyes heavy":
        if current_heavy_eye_seconds >= EYE_CLOSURE_HIGH_RISK_SECONDS:
            eye_evidence = max(eye_evidence, 0.45)
        elif current_heavy_eye_seconds >= DROWSINESS_SECONDS_THRESHOLD:
            eye_evidence = max(eye_evidence, 0.34)
        else:
            eye_evidence = max(eye_evidence, 0.20)
    elif eye_status == "Eyes closed":
        if current_eye_closed_seconds >= DROWSINESS_SECONDS_THRESHOLD:
            eye_evidence = max(eye_evidence, 0.52)
        else:
            eye_evidence = max(eye_evidence, 0.35)
    elif eye_status == "Drowsiness warning":
        if current_eye_closed_seconds >= EYE_CLOSURE_CRITICAL_SECONDS:
            eye_evidence = max(eye_evidence, 0.92)
        elif current_eye_closed_seconds >= EYE_CLOSURE_HIGH_RISK_SECONDS:
            eye_evidence = max(eye_evidence, 0.74)
        else:
            eye_evidence = max(eye_evidence, 0.58)

    face_visible_duration = number_or_none(
        feature_values.get("temporal_60s_face_visible_duration")
    )
    temporal_ready = (
        face_visible_duration is not None
        and face_visible_duration >= TEMPORAL_SCORING_MIN_SECONDS
    )
    perclos_percent = number_or_none(
        feature_values.get("temporal_30s_perclos_percent")
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
    seconds_since_last_microsleep = first_number_or_none(
        feature_values.get("seconds_since_last_microsleep"),
        feature_values.get("temporal_60s_seconds_since_last_microsleep"),
    )

    if microsleep_active or microsleep_recent:
        if current_eye_closed_seconds >= EYE_CLOSURE_CRITICAL_SECONDS:
            eye_evidence = max(eye_evidence, 0.95)
        elif current_eye_closed_seconds >= EYE_CLOSURE_HIGH_RISK_SECONDS:
            eye_evidence = max(eye_evidence, 0.78)
        else:
            eye_evidence = max(eye_evidence, 0.62)
    elif (
        temporal_ready
        and (
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
        mouth_evidence = max(min(mouth_evidence, 0.45), 0.35)
    else:
        mouth_evidence = min(mouth_evidence, 0.20)

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

    longest_head_down_duration = number_or_none(
        feature_values.get("temporal_30s_longest_head_down_duration")
    )
    longest_side_look_duration = number_or_none(
        feature_values.get("temporal_30s_longest_side_look_duration")
    )
    head_pose_instability = number_or_none(
        feature_values.get("temporal_30s_head_pose_instability")
    )

    head_evidence = 0.0
    if (
        longest_head_down_duration is not None
        and longest_head_down_duration >= PROLONGED_HEAD_DOWN_SECONDS
    ):
        head_evidence = max(head_evidence, 0.55)

    if (
        longest_side_look_duration is not None
        and longest_side_look_duration >= PROLONGED_SIDE_LOOK_SECONDS
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
    if driver_status == "Driver critically drowsy":
        status_evidence = 0.85
    elif driver_status == "Driver drowsy":
        status_evidence = 0.60
    elif driver_status == "Driver attention required":
        status_evidence = 0.35

    score_evidence = clamp(score / 100.0)
    has_recent_microsleep = (
        seconds_since_last_microsleep is not None
        and seconds_since_last_microsleep <= 8.0
    )
    has_recent_yawn = (
        seconds_since_last_yawn is not None
        and seconds_since_last_yawn <= 8.0
    )

    evidence = max(
        eye_evidence,
        mouth_evidence,
        head_evidence,
        status_evidence,
        score_evidence,
    )
    is_instant_neutral = (
        eye_status == "Eyes open"
        and mouth_status == "Mouth normal"
        and head_status == "Head normal"
    )
    has_repeated_yawns = (
        (yawn_count_120s is not None and yawn_count_120s >= FREQUENT_YAWN_COUNT_THRESHOLD)
        or (yawn_count_60s is not None and yawn_count_60s >= 2)
    )
    has_strong_temporal_eye_signal = (
        (perclos_percent is not None and perclos_percent >= PERCLOS_HIGH_PERCENT)
        and slow_blink_count is not None
        and slow_blink_count >= SLOW_BLINK_COUNT_THRESHOLD
    )

    if is_instant_neutral and not (
        microsleep_active
        or microsleep_recent
        or has_recent_microsleep
        or has_recent_yawn
        or has_repeated_yawns
        or has_strong_temporal_eye_signal
    ):
        evidence = min(evidence, 0.20)
    elif is_instant_neutral:
        if has_recent_microsleep:
            evidence = min(evidence, 0.45)
        elif has_repeated_yawns:
            evidence = min(evidence, 0.50)
        else:
            evidence = min(evidence, 0.35)

    return evidence


def calibrate_live_probability(raw_drowsy_probability, live_evidence):
    if raw_drowsy_probability is None or live_evidence is None:
        return raw_drowsy_probability

    if live_evidence < 0.15:
        return min(raw_drowsy_probability, 0.18)

    if live_evidence < 0.35:
        calibrated = (raw_drowsy_probability * 0.30) + (live_evidence * 0.70)
        return min(calibrated, 0.32)

    if live_evidence < 0.55:
        calibrated = (raw_drowsy_probability * 0.35) + (live_evidence * 0.65)
        return min(calibrated, 0.48)

    if live_evidence < 0.75:
        calibrated = (raw_drowsy_probability * 0.40) + (live_evidence * 0.60)
        return min(calibrated, 0.65)

    calibrated = (raw_drowsy_probability * 0.35) + (live_evidence * 0.65)
    if live_evidence >= 0.90:
        return max(clamp(calibrated), 0.85)

    return clamp(calibrated)


def get_neutral_recovery_cap(feature_values, live_context):
    if not live_context:
        return None

    is_currently_neutral = (
        live_context.get("eye_status") == "Eyes open"
        and live_context.get("mouth_status") == "Mouth normal"
        and live_context.get("head_status") == "Head normal"
    )
    if not is_currently_neutral:
        return None

    seconds_since_last_microsleep = first_number_or_none(
        feature_values.get("seconds_since_last_microsleep"),
        feature_values.get("temporal_60s_seconds_since_last_microsleep"),
    )
    seconds_since_last_yawn = number_or_none(
        feature_values.get("seconds_since_last_yawn")
    )
    yawn_count_60s = number_or_none(feature_values.get("yawn_count_60s"))
    yawn_count_120s = number_or_none(feature_values.get("yawn_count_120s"))
    perclos_percent = number_or_none(
        feature_values.get("temporal_30s_perclos_percent")
    )
    slow_blink_count = number_or_none(
        feature_values.get("temporal_60s_slow_blink_count")
    )

    has_repeated_yawns = (
        (yawn_count_120s is not None and yawn_count_120s >= FREQUENT_YAWN_COUNT_THRESHOLD)
        or (yawn_count_60s is not None and yawn_count_60s >= 2)
    )
    if has_repeated_yawns:
        return 0.50

    if seconds_since_last_yawn is not None and seconds_since_last_yawn <= 8.0:
        return 0.35

    if seconds_since_last_microsleep is not None:
        if seconds_since_last_microsleep <= 3.0:
            return 0.60
        if seconds_since_last_microsleep <= 8.0:
            return 0.45
        if seconds_since_last_microsleep <= 15.0:
            return 0.35

    has_stale_temporal_eye_signal = (
        perclos_percent is not None
        and perclos_percent >= PERCLOS_HIGH_PERCENT
        and slow_blink_count is not None
        and slow_blink_count >= SLOW_BLINK_COUNT_THRESHOLD
    )
    if has_stale_temporal_eye_signal:
        return 0.35

    return 0.28


def get_transient_eye_closure_cap(feature_values, live_context):
    if not live_context:
        return None

    eye_status = live_context.get("eye_status")
    if eye_status not in {"Eyes heavy", "Eyes closed", "Drowsiness warning"}:
        return None

    current_heavy_eye_seconds = number_or_none(
        feature_values.get(
            "current_heavy_eye_seconds",
            live_context.get("current_heavy_eye_seconds"),
        )
    )
    if eye_status == "Eyes heavy":
        current_heavy_eye_seconds = current_heavy_eye_seconds or 0.0
        if current_heavy_eye_seconds < DROWSINESS_SECONDS_THRESHOLD:
            return 0.35
        if current_heavy_eye_seconds < EYE_CLOSURE_HIGH_RISK_SECONDS:
            return 0.50
        if current_heavy_eye_seconds < EYE_CLOSURE_CRITICAL_SECONDS:
            return 0.65
        return None

    current_eye_closed_seconds = number_or_none(
        feature_values.get(
            "current_eye_closed_seconds",
            live_context.get("current_eye_closed_seconds"),
        )
    )
    if current_eye_closed_seconds is None:
        return None

    if current_eye_closed_seconds <= NORMAL_BLINK_GRACE_SECONDS:
        return NORMAL_BLINK_RISK_CAP

    if current_eye_closed_seconds < DROWSINESS_SECONDS_THRESHOLD:
        return SHORT_EYE_CLOSURE_RISK_CAP

    if current_eye_closed_seconds < EYE_CLOSURE_HIGH_RISK_SECONDS:
        return 0.60

    if current_eye_closed_seconds < EYE_CLOSURE_CRITICAL_SECONDS:
        return 0.78

    return None


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
    transient_cap = get_transient_eye_closure_cap(feature_values, live_context)
    if transient_cap is not None:
        drowsy_probability = min(drowsy_probability, transient_cap)
    neutral_recovery_cap = get_neutral_recovery_cap(feature_values, live_context)
    if neutral_recovery_cap is not None:
        drowsy_probability = min(drowsy_probability, neutral_recovery_cap)

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
