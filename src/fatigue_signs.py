from dataclasses import dataclass

from config import (
    BLINK_RATE_MIN_OBSERVATION_SECONDS,
    FREQUENT_BLINKS_PER_MINUTE_THRESHOLD,
    FREQUENT_YAWN_COUNT_THRESHOLD,
    FREQUENT_YAWN_WINDOW_SECONDS,
    HEAD_POSE_INSTABILITY_THRESHOLD,
    HEAVY_EYE_EAR_RATIO,
    HEAVY_EYE_SECONDS_THRESHOLD,
    LONG_YAWN_SECONDS_THRESHOLD,
    MICROSLEEP_SECONDS_THRESHOLD,
    PROLONGED_HEAD_DOWN_SECONDS,
    PROLONGED_SIDE_LOOK_SECONDS,
    SLOW_BLINK_COUNT_THRESHOLD,
    SLOW_BLINK_SECONDS_THRESHOLD,
    TEMPORAL_SCORING_MIN_SECONDS,
)


EYES_CLOSED = "eyes_closed"
HEAVY_EYES = "heavy_eyes"
FREQUENT_BLINKING = "frequent_blinking"
SLOW_BLINKING = "slow_blinking"
MICROSLEEP = "microsleep"
FREQUENT_YAWNING = "frequent_yawning"
HEAD_DOWN = "head_down"
PROLONGED_SIDE_LOOK = "prolonged_side_look"
UNSTABLE_HEAD_POSE = "unstable_head_pose"


@dataclass(frozen=True)
class FatigueSignDefinition:
    sign_id: str
    label: str
    description: str
    category: str
    default_threshold: str
    requires_temporal_analysis: bool
    severity_weight: int


FATIGUE_SIGN_DEFINITIONS = {
    EYES_CLOSED: FatigueSignDefinition(
        sign_id=EYES_CLOSED,
        label="Eyes closed",
        description="Both eyes are below the adaptive closed-eye EAR threshold.",
        category="eyes",
        default_threshold="EAR < adaptive threshold",
        requires_temporal_analysis=False,
        severity_weight=25,
    ),
    HEAVY_EYES: FatigueSignDefinition(
        sign_id=HEAVY_EYES,
        label="Heavy eyes",
        description=(
            "Both eyes remain partially closed for longer than a normal blink, "
            "but are not fully closed."
        ),
        category="eyes",
        default_threshold=(
            f"EAR < adaptive threshold * {HEAVY_EYE_EAR_RATIO:.2f} for "
            f"{HEAVY_EYE_SECONDS_THRESHOLD:.2f}s"
        ),
        requires_temporal_analysis=False,
        severity_weight=15,
    ),
    FREQUENT_BLINKING: FatigueSignDefinition(
        sign_id=FREQUENT_BLINKING,
        label="Frequent blinking",
        description="Blink frequency is higher than normal within the active time window.",
        category="eyes",
        default_threshold=(
            f">= {FREQUENT_BLINKS_PER_MINUTE_THRESHOLD} blinks/minute after "
            f"{BLINK_RATE_MIN_OBSERVATION_SECONDS:.0f}s"
        ),
        requires_temporal_analysis=True,
        severity_weight=15,
    ),
    SLOW_BLINKING: FatigueSignDefinition(
        sign_id=SLOW_BLINKING,
        label="Slow blinking",
        description="Eye closures last longer than normal blinks.",
        category="eyes",
        default_threshold=(
            f">= {SLOW_BLINK_COUNT_THRESHOLD} blinks longer than "
            f"{SLOW_BLINK_SECONDS_THRESHOLD:.2f}s"
        ),
        requires_temporal_analysis=True,
        severity_weight=20,
    ),
    MICROSLEEP: FatigueSignDefinition(
        sign_id=MICROSLEEP,
        label="Microsleep",
        description="Both eyes remain closed long enough to indicate a microsleep event.",
        category="eyes",
        default_threshold=f">= {MICROSLEEP_SECONDS_THRESHOLD:.1f}s closed eyes",
        requires_temporal_analysis=True,
        severity_weight=45,
    ),
    FREQUENT_YAWNING: FatigueSignDefinition(
        sign_id=FREQUENT_YAWNING,
        label="Frequent yawning",
        description="Yawning appears repeatedly in a short time interval.",
        category="mouth",
        default_threshold=(
            f">= {FREQUENT_YAWN_COUNT_THRESHOLD} yawns / "
            f"{FREQUENT_YAWN_WINDOW_SECONDS}s"
        ),
        requires_temporal_analysis=True,
        severity_weight=25,
    ),
    HEAD_DOWN: FatigueSignDefinition(
        sign_id=HEAD_DOWN,
        label="Head down",
        description="Head pitch indicates that the driver is looking down.",
        category="head_pose",
        default_threshold=f">= {PROLONGED_HEAD_DOWN_SECONDS:.1f}s head down",
        requires_temporal_analysis=True,
        severity_weight=30,
    ),
    PROLONGED_SIDE_LOOK: FatigueSignDefinition(
        sign_id=PROLONGED_SIDE_LOOK,
        label="Prolonged side look",
        description="The driver looks sideways for too long.",
        category="head_pose",
        default_threshold=f">= {PROLONGED_SIDE_LOOK_SECONDS:.1f}s side look",
        requires_temporal_analysis=True,
        severity_weight=25,
    ),
    UNSTABLE_HEAD_POSE: FatigueSignDefinition(
        sign_id=UNSTABLE_HEAD_POSE,
        label="Unstable head pose",
        description="Pitch, yaw and roll vary strongly within the monitoring window.",
        category="head_pose",
        default_threshold=f"instability >= {HEAD_POSE_INSTABILITY_THRESHOLD:.1f} deg",
        requires_temporal_analysis=True,
        severity_weight=20,
    ),
}


def get_fatigue_sign_definition(sign_id):
    return FATIGUE_SIGN_DEFINITIONS[sign_id]


def list_fatigue_signs():
    return list(FATIGUE_SIGN_DEFINITIONS.values())


def classify_instant_fatigue_signs(eye_status, mouth_status, head_status):
    signs = []

    if eye_status == "Eyes closed":
        signs.append(EYES_CLOSED)
    elif eye_status == "Eyes heavy":
        signs.append(HEAVY_EYES)
    elif eye_status == "Drowsiness warning":
        signs.append(EYES_CLOSED)

    return signs


def classify_temporal_fatigue_signs(temporal_features, window_seconds=60):
    if not temporal_features:
        return []

    prefix = f"temporal_{window_seconds}s"
    signs = []
    blink_rate = _number_or_none(
        temporal_features.get(f"{prefix}_blink_rate_per_minute")
    )
    face_visible_duration = _number_or_zero(
        temporal_features.get(f"{prefix}_face_visible_duration")
    )
    temporal_ready = face_visible_duration >= TEMPORAL_SCORING_MIN_SECONDS
    avg_blink_duration = _number_or_none(
        temporal_features.get(f"{prefix}_avg_blink_duration")
    )
    slow_blink_count = _number_or_zero(
        temporal_features.get(f"{prefix}_slow_blink_count")
    )
    microsleep_event_count = _number_or_zero(
        temporal_features.get(f"{prefix}_microsleep_event_count")
    )
    microsleep_active = bool(temporal_features.get(f"{prefix}_microsleep_active"))
    microsleep_recent = bool(temporal_features.get(f"{prefix}_microsleep_recent"))
    yawn_count_60s = _number_or_zero(temporal_features.get("yawn_count_60s"))
    yawn_count_120s = _number_or_zero(temporal_features.get("yawn_count_120s"))
    head_prefix = "temporal_30s"
    longest_head_down_duration = _number_or_zero(
        temporal_features.get(f"{head_prefix}_longest_head_down_duration")
    )
    longest_side_look_duration = _number_or_zero(
        temporal_features.get(f"{head_prefix}_longest_side_look_duration")
    )
    head_pose_instability = _number_or_none(
        temporal_features.get(f"{head_prefix}_head_pose_instability")
    )

    if (
        temporal_ready
        and blink_rate is not None
        and face_visible_duration >= BLINK_RATE_MIN_OBSERVATION_SECONDS
        and blink_rate >= FREQUENT_BLINKS_PER_MINUTE_THRESHOLD
    ):
        signs.append(FREQUENT_BLINKING)

    has_microsleep = (
        microsleep_active
        or microsleep_recent
    )
    if has_microsleep:
        signs.append(MICROSLEEP)

    slow_blinks_without_microsleep = max(0, slow_blink_count - microsleep_event_count)
    if (
        temporal_ready
        and (
            slow_blinks_without_microsleep >= SLOW_BLINK_COUNT_THRESHOLD
            or (
                avg_blink_duration is not None
                and avg_blink_duration >= SLOW_BLINK_SECONDS_THRESHOLD
                and slow_blink_count >= SLOW_BLINK_COUNT_THRESHOLD
                and not has_microsleep
            )
        )
    ):
        signs.append(SLOW_BLINKING)

    if (
        yawn_count_120s >= FREQUENT_YAWN_COUNT_THRESHOLD
        or yawn_count_60s >= max(2, FREQUENT_YAWN_COUNT_THRESHOLD - 1)
    ):
        signs.append(FREQUENT_YAWNING)

    if longest_head_down_duration >= PROLONGED_HEAD_DOWN_SECONDS:
        signs.append(HEAD_DOWN)

    if longest_side_look_duration >= PROLONGED_SIDE_LOOK_SECONDS:
        signs.append(PROLONGED_SIDE_LOOK)

    if (
        head_pose_instability is not None
        and head_pose_instability >= HEAD_POSE_INSTABILITY_THRESHOLD
    ):
        signs.append(UNSTABLE_HEAD_POSE)

    return signs


def merge_fatigue_signs(*sign_groups):
    merged = []

    for sign_group in sign_groups:
        for sign_id in sign_group:
            if sign_id not in merged:
                merged.append(sign_id)

    return merged


def format_sign_labels(sign_ids):
    return [
        FATIGUE_SIGN_DEFINITIONS[sign_id].label
        for sign_id in sign_ids
        if sign_id in FATIGUE_SIGN_DEFINITIONS
    ]


def _number_or_none(value):
    if value is None:
        return None

    return float(value)


def _number_or_zero(value):
    if value is None:
        return 0.0

    return float(value)
