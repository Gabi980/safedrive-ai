import joblib
import pandas as pd

from config import ML_DROWSY_PROBABILITY_THRESHOLD, ML_MODEL_PATH, YAWN_MAR_THRESHOLD


ABNORMAL_HEAD_STATUSES = {"Head down", "Looking sideways", "Head tilted"}


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def unavailable_result(model_name=None):
    return {
        "ml_model_name": model_name,
        "ml_prediction": "Unavailable",
        "ml_confidence": None,
        "ml_drowsy_probability": None,
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

    if eye_status == "Eyes closed":
        eye_evidence = 0.35
    elif eye_status == "Drowsiness warning":
        eye_evidence = max(eye_evidence, 0.92)

    mouth_evidence = 0.0
    if mar is not None and mar_threshold:
        mouth_start = mar_threshold * 0.75
        mouth_range = mar_threshold - mouth_start
        mouth_evidence = clamp((mar - mouth_start) / mouth_range)

    if mouth_status == "Yawning detected":
        mouth_evidence = max(mouth_evidence, 0.78)
    else:
        mouth_evidence = min(mouth_evidence, 0.25)

    head_evidence = 0.35 if head_status in ABNORMAL_HEAD_STATUSES else 0.0

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
        return min(raw_drowsy_probability, 0.30)

    if live_evidence < 0.40:
        calibrated = (raw_drowsy_probability * 0.25) + (live_evidence * 0.75)
        return min(calibrated, 0.45)

    calibrated = (raw_drowsy_probability * 0.35) + (live_evidence * 0.65)
    return clamp(max(calibrated, live_evidence))


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
        "ml_calibrated_drowsy_probability": drowsy_probability,
        "ml_raw_drowsy_probability": raw_drowsy_probability,
        "ml_alert_probability": alert_probability,
        "ml_raw_alert_probability": raw_alert_probability,
        "ml_live_evidence": live_evidence,
        "ml_drowsy_threshold": ML_DROWSY_PROBABILITY_THRESHOLD,
    }
