import time
import winsound

from config import (
    AUDIO_ALERT_COOLDOWN_SECONDS,
    FREQUENT_YAWN_COUNT_THRESHOLD,
    HEAD_POSE_INSTABILITY_THRESHOLD,
    LONG_YAWN_SECONDS_THRESHOLD,
    PERCLOS_HIGH_PERCENT,
    PERCLOS_WARNING_PERCENT,
    PROLONGED_HEAD_DOWN_SECONDS,
    PROLONGED_SIDE_LOOK_SECONDS,
    SLOW_BLINK_COUNT_THRESHOLD,
)
from fatigue_signs import (
    FREQUENT_BLINKING,
    FREQUENT_YAWNING,
    HEAD_DOWN,
    HEAVY_EYES,
    MICROSLEEP,
    PROLONGED_SIDE_LOOK,
    SLOW_BLINKING,
    UNSTABLE_HEAD_POSE,
)


def get_rule_based_driver_status(
    eye_status,
    mouth_status,
    head_status,
    score=None,
    warning_signs=None,
):
    if score is None or warning_signs is None:
        score, warning_signs = calculate_rule_based_score(
            eye_status,
            mouth_status,
            head_status,
        )

    if score >= 80:
        return "Driver critically drowsy", (0, 0, 255), warning_signs

    if score >= 60:
        return "Driver drowsy", (0, 80, 255), warning_signs

    if score >= 30:
        return "Driver attention required", (0, 165, 255), warning_signs

    return "Driver OK", (0, 255, 0), warning_signs


def calculate_drowsiness_score(eye_status, mouth_status, head_status):
    score, _ = calculate_rule_based_score(eye_status, mouth_status, head_status)
    return score


def calculate_rule_based_score(
    eye_status,
    mouth_status,
    head_status,
    temporal_features=None,
    fatigue_sign_ids=None,
):
    temporal_features = temporal_features or {}
    fatigue_sign_ids = set(fatigue_sign_ids or [])
    score = 0
    warning_signs = []

    def add(points, reason):
        nonlocal score
        score += points
        add_reason(reason)

    def ensure_at_least(value, reason):
        nonlocal score
        if score < value:
            score = value
        add_reason(reason)

    def add_reason(reason):
        if reason not in warning_signs:
            warning_signs.append(reason)

    perclos_percent = _number_or_zero(
        temporal_features.get("temporal_60s_perclos_percent")
    )
    slow_blink_count = _number_or_zero(temporal_features.get("temporal_60s_slow_blink_count"))
    microsleep_active = bool(temporal_features.get("temporal_60s_microsleep_active"))
    microsleep_recent = bool(temporal_features.get("temporal_60s_microsleep_recent"))
    yawn_count_60s = _number_or_zero(temporal_features.get("yawn_count_60s"))
    yawn_count_120s = _number_or_zero(temporal_features.get("yawn_count_120s"))
    avg_yawn_duration = _number_or_zero(temporal_features.get("avg_yawn_duration"))
    head_down_duration = _number_or_zero(
        temporal_features.get("temporal_30s_head_down_duration")
    )
    side_look_duration = _number_or_zero(
        temporal_features.get("temporal_30s_side_look_duration")
    )
    head_pose_instability = _number_or_zero(
        temporal_features.get("temporal_30s_head_pose_instability")
    )

    if eye_status == "Eyes heavy":
        add(15, "heavy_eyes")
    elif eye_status == "Eyes closed":
        add(30, "eyes_closed")
    elif eye_status == "Drowsiness warning":
        add(55, "eyes_closed_long")

    if mouth_status == "Yawning detected":
        add(12, "current_yawn")

    if perclos_percent >= PERCLOS_HIGH_PERCENT:
        add(30, "high_perclos")
    elif perclos_percent >= PERCLOS_WARNING_PERCENT:
        add(18, "elevated_perclos")

    has_repeated_slow_blinks = (
        SLOW_BLINKING in fatigue_sign_ids
        or slow_blink_count >= SLOW_BLINK_COUNT_THRESHOLD
    )
    if has_repeated_slow_blinks:
        add(18, "slow_blinking")

    has_microsleep = (
        MICROSLEEP in fatigue_sign_ids
        or microsleep_active
        or microsleep_recent
    )
    if has_microsleep:
        ensure_at_least(90, "microsleep")

    repeated_yawning = (
        FREQUENT_YAWNING in fatigue_sign_ids
        or yawn_count_120s >= FREQUENT_YAWN_COUNT_THRESHOLD
        or yawn_count_60s >= max(2, FREQUENT_YAWN_COUNT_THRESHOLD - 1)
    )
    if repeated_yawning:
        add(25, "repeated_yawning")

    if avg_yawn_duration >= LONG_YAWN_SECONDS_THRESHOLD:
        add(10, "long_yawn")

    head_down_too_long = (
        HEAD_DOWN in fatigue_sign_ids
        or head_down_duration >= PROLONGED_HEAD_DOWN_SECONDS
    )
    if head_down_too_long:
        add(30, "prolonged_head_down")

    if (
        PROLONGED_SIDE_LOOK in fatigue_sign_ids
        or side_look_duration >= PROLONGED_SIDE_LOOK_SECONDS
    ):
        add(18, "prolonged_side_look")

    if (
        UNSTABLE_HEAD_POSE in fatigue_sign_ids
        or head_pose_instability >= HEAD_POSE_INSTABILITY_THRESHOLD
    ):
        add(12, "unstable_head_pose")

    if FREQUENT_BLINKING in fatigue_sign_ids:
        add(10, "frequent_blinking")

    if perclos_percent >= PERCLOS_HIGH_PERCENT and has_repeated_slow_blinks:
        ensure_at_least(65, "high_perclos_with_slow_blinks")

    has_heavy_eyes = eye_status == "Eyes heavy" or HEAVY_EYES in fatigue_sign_ids
    if repeated_yawning and has_heavy_eyes:
        ensure_at_least(75, "repeated_yawning_with_heavy_eyes")

    eyes_closed_now = eye_status in {"Eyes closed", "Drowsiness warning"}
    if head_down_too_long and eyes_closed_now:
        ensure_at_least(90, "head_down_with_closed_eyes")

    if len(warning_signs) >= 3 and score < 70:
        ensure_at_least(70, "multiple_fatigue_signs")

    return min(score, 100), warning_signs


def _number_or_zero(value):
    if value is None:
        return 0.0

    return float(value)


def get_alert_level(score):
    if score < 30:
        return "Level 0: OK", (0, 255, 0), 0

    if score < 60:
        return "Level 1: Visual warning", (0, 165, 255), 1

    if score < 80:
        return "Level 2: Audio alert", (0, 80, 255), 2

    return "Level 3: Critical alert", (0, 0, 255), 3


def play_alert_sound(alert_level, last_sound_time):
    if alert_level < 2:
        return last_sound_time

    current_time = time.time()
    if current_time - last_sound_time < AUDIO_ALERT_COOLDOWN_SECONDS:
        return last_sound_time

    if alert_level == 2:
        winsound.Beep(1200, 180)
    else:
        winsound.Beep(1800, 180)
        winsound.Beep(1800, 180)

    return current_time
