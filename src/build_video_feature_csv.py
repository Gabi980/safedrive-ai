import argparse
import csv
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from alert_system import calculate_rule_based_score, get_alert_level
from config import (
    DROWSINESS_SECONDS_THRESHOLD,
    EYE_CLOSED_CONFIRMATION_RATIO,
    EYE_CLOSED_EAR_THRESHOLD,
    HEAVY_EYE_EAR_RATIO,
    HEAVY_EYE_SECONDS_THRESHOLD,
    MODEL_PATH,
    YAWN_MAR_THRESHOLD,
)
from facial_features import calculate_ear, calculate_mar, get_mouth_status
from fatigue_signs import (
    HEAD_DOWN,
    PROLONGED_SIDE_LOOK,
    UNSTABLE_HEAD_POSE,
    classify_instant_fatigue_signs,
    classify_temporal_fatigue_signs,
    format_sign_labels,
    merge_fatigue_signs,
)
from head_pose import estimate_head_pose, get_head_status
from landmark_indexes import LEFT_EAR_POINTS, MOUTH_MAR_POINTS, RIGHT_EAR_POINTS
from temporal_analysis import TemporalFatigueAnalyzer


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_LOG_EVERY = 300
ABNORMAL_HEAD_SIGN_IDS = {HEAD_DOWN, PROLONGED_SIDE_LOOK, UNSTABLE_HEAD_POSE}

TEMPORAL_FEATURE_COLUMNS = [
    "temporal_10s_face_visible_duration",
    "temporal_10s_perclos_percent",
    "temporal_10s_closed_eye_ratio",
    "temporal_10s_heavy_eye_ratio",
    "temporal_10s_blink_rate_per_minute",
    "temporal_10s_avg_blink_duration",
    "temporal_10s_longest_eye_closure",
    "temporal_10s_slow_blink_count",
    "temporal_30s_face_visible_duration",
    "temporal_30s_perclos_percent",
    "temporal_30s_closed_eye_ratio",
    "temporal_30s_heavy_eye_ratio",
    "temporal_30s_blink_rate_per_minute",
    "temporal_30s_avg_blink_duration",
    "temporal_30s_longest_eye_closure",
    "temporal_30s_slow_blink_count",
    "temporal_30s_head_down_duration",
    "temporal_30s_side_look_duration",
    "temporal_30s_head_abnormal_duration",
    "temporal_30s_longest_head_down_duration",
    "temporal_30s_longest_side_look_duration",
    "temporal_30s_head_pose_instability",
    "temporal_60s_face_visible_duration",
    "temporal_60s_perclos_percent",
    "temporal_60s_closed_eye_ratio",
    "temporal_60s_heavy_eye_ratio",
    "temporal_60s_blink_count",
    "temporal_60s_blink_rate_per_minute",
    "temporal_60s_avg_blink_duration",
    "temporal_60s_longest_eye_closure",
    "temporal_60s_slow_blink_count",
    "temporal_60s_microsleep_event_count",
    "temporal_60s_microsleep_active",
    "temporal_60s_microsleep_recent",
    "temporal_60s_head_down_duration",
    "temporal_60s_side_look_duration",
    "temporal_60s_head_abnormal_duration",
    "temporal_60s_longest_head_down_duration",
    "temporal_60s_longest_side_look_duration",
    "temporal_60s_head_pose_instability",
    "yawn_count_60s",
    "yawn_count_120s",
    "avg_yawn_duration_60s",
    "avg_yawn_duration_120s",
    "max_yawn_duration",
    "long_yawn_count",
]

CSV_COLUMNS = [
    "source_dataset",
    "source_type",
    "video_id",
    "video_path",
    "participant_id",
    "original_label",
    "original_label_name",
    "label",
    "label_name",
    "frame_index",
    "timestamp_seconds",
    "fps",
    "face_detected",
    "usable_for_training",
    "ear",
    "left_ear",
    "right_ear",
    "ear_threshold",
    "mar",
    "mar_threshold",
    "pitch",
    "yaw",
    "roll",
    "eye_closed_signal",
    "yawn_signal",
    "head_abnormal_signal",
    "current_eye_closed_seconds",
    "current_heavy_eye_seconds",
    "score",
    "alert_level",
    "eye_status",
    "mouth_status",
    "head_status",
    "fatigue_signs",
    *TEMPORAL_FEATURE_COLUMNS,
]


class VideoEyeState:
    def __init__(self):
        self.closed_start_time = None
        self.heavy_start_time = None

    def update(self, ear, timestamp_seconds, ear_threshold=EYE_CLOSED_EAR_THRESHOLD):
        heavy_eye_threshold = ear_threshold * HEAVY_EYE_EAR_RATIO
        closed_eye_threshold = ear_threshold * EYE_CLOSED_CONFIRMATION_RATIO

        if ear >= heavy_eye_threshold:
            self.closed_start_time = None
            self.heavy_start_time = None
            return "Eyes open", 0.0, 0.0

        if ear >= closed_eye_threshold:
            self.closed_start_time = None
            if self.heavy_start_time is None:
                self.heavy_start_time = timestamp_seconds

            heavy_seconds = timestamp_seconds - self.heavy_start_time
            if heavy_seconds >= HEAVY_EYE_SECONDS_THRESHOLD:
                return "Eyes heavy", 0.0, heavy_seconds

            return "Eyes open", 0.0, heavy_seconds

        self.heavy_start_time = None
        if self.closed_start_time is None:
            self.closed_start_time = timestamp_seconds

        closed_seconds = timestamp_seconds - self.closed_start_time
        if closed_seconds >= DROWSINESS_SECONDS_THRESHOLD:
            return "Drowsiness warning", closed_seconds, 0.0

        return "Eyes closed", closed_seconds, 0.0


def create_video_landmarker():
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


def list_videos(video_root):
    return sorted(
        path
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def video_label_from_path(video_path):
    label_value = int(video_path.stem)

    if label_value == 0:
        return label_value, "alert", 0, "alert"

    if label_value == 5:
        return label_value, "low_vigilant", 1, "drowsy"

    if label_value == 10:
        return label_value, "drowsy", 1, "drowsy"

    raise ValueError(f"Unknown video label in file name: {video_path.name}")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    if hours:
        return f"{hours}h {minutes}m {remaining_seconds}s"

    if minutes:
        return f"{minutes}m {remaining_seconds}s"

    return f"{remaining_seconds}s"


def normalize_value(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    return value


def empty_temporal_features():
    return {column: None for column in TEMPORAL_FEATURE_COLUMNS}


def write_row(writer, base_row, temporal_features):
    row = {**base_row, **empty_temporal_features()}

    for column in TEMPORAL_FEATURE_COLUMNS:
        row[column] = normalize_value(temporal_features.get(column))

    writer.writerow(row)


def print_video_progress(
    video_path,
    processed_frames,
    total_to_process,
    detected_faces,
    started_at,
):
    elapsed = time.time() - started_at
    speed = processed_frames / elapsed if elapsed > 0 else 0.0
    remaining = total_to_process - processed_frames
    eta = remaining / speed if speed > 0 else 0.0
    percent = processed_frames / total_to_process * 100 if total_to_process else 100.0

    print(
        f"[{video_path.parent.name}/{video_path.name}] "
        f"{processed_frames}/{total_to_process} ({percent:.1f}%) | "
        f"faces: {detected_faces} | speed: {speed:.1f} frames/s | "
        f"elapsed: {format_duration(elapsed)} | ETA: {format_duration(eta)}",
        flush=True,
    )


def process_video(video_path, writer, args):
    original_label, original_label_name, label, label_name = video_label_from_path(video_path)
    participant_id = video_path.parent.name
    video_id = f"{participant_id}_{original_label}"
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[{video_id}] skipped, cannot open video: {video_path}", flush=True)
        return 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = args.fallback_fps

    if args.sample_fps > 0:
        frame_step = max(1, round(fps / args.sample_fps))
    else:
        frame_step = max(1, args.frame_step)

    total_to_process = frame_count // frame_step
    if frame_count % frame_step:
        total_to_process += 1
    if args.max_frames_per_video:
        total_to_process = min(total_to_process, args.max_frames_per_video)

    print(
        f"\n[{video_id}] {video_path} | label={original_label_name} "
        f"binary={label_name} | fps={fps:.2f} | frames={frame_count} | "
        f"step={frame_step} | rows={total_to_process}",
        flush=True,
    )

    analyzer = TemporalFatigueAnalyzer()
    eye_state = VideoEyeState()
    landmarker = create_video_landmarker()
    processed_frames = 0
    detected_faces = 0
    frame_index = -1
    started_at = time.time()
    last_timestamp_ms = -1

    try:
        while True:
            grabbed = cap.grab()
            if not grabbed:
                break

            frame_index += 1
            if frame_index % frame_step != 0:
                continue

            if args.max_frames_per_video and processed_frames >= args.max_frames_per_video:
                break

            timestamp_seconds = frame_index / fps
            timestamp_ms = int(timestamp_seconds * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            success, frame = cap.retrieve()
            if not success:
                break

            row = build_base_row(
                video_path,
                video_id,
                participant_id,
                original_label,
                original_label_name,
                label,
                label_name,
                frame_index,
                timestamp_seconds,
                fps,
            )
            temporal_features = {}

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame),
                    timestamp_ms,
                )

                if result.face_landmarks:
                    detected_faces += 1
                    row, temporal_features = process_detected_face(
                        frame,
                        result.face_landmarks[0],
                        analyzer,
                        eye_state,
                        row,
                        timestamp_seconds,
                    )
                else:
                    temporal_features = analyzer.add_sample(
                        timestamp=timestamp_seconds,
                        face_detected=False,
                        eye_status="Unavailable",
                        mouth_status="Unavailable",
                        head_status="Unavailable",
                    )
            except Exception as error:
                row["processing_error"] = str(error)
                temporal_features = analyzer.add_sample(
                    timestamp=timestamp_seconds,
                    face_detected=False,
                    eye_status="Unavailable",
                    mouth_status="Unavailable",
                    head_status="Unavailable",
                )

            write_row(writer, row, temporal_features)
            processed_frames += 1

            if (
                processed_frames == 1
                or processed_frames % args.log_every == 0
                or processed_frames == total_to_process
            ):
                print_video_progress(
                    video_path,
                    processed_frames,
                    total_to_process,
                    detected_faces,
                    started_at,
                )
    finally:
        landmarker.close()
        cap.release()

    return processed_frames, detected_faces


def build_base_row(
    video_path,
    video_id,
    participant_id,
    original_label,
    original_label_name,
    label,
    label_name,
    frame_index,
    timestamp_seconds,
    fps,
):
    return {
        "source_dataset": "video_temporal_dataset",
        "source_type": "video_frame",
        "video_id": video_id,
        "video_path": str(video_path),
        "participant_id": participant_id,
        "original_label": original_label,
        "original_label_name": original_label_name,
        "label": label,
        "label_name": label_name,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "fps": fps,
        "face_detected": False,
        "usable_for_training": False,
        "ear": None,
        "left_ear": None,
        "right_ear": None,
        "ear_threshold": EYE_CLOSED_EAR_THRESHOLD,
        "mar": None,
        "mar_threshold": YAWN_MAR_THRESHOLD,
        "pitch": None,
        "yaw": None,
        "roll": None,
        "eye_closed_signal": None,
        "yawn_signal": None,
        "head_abnormal_signal": None,
        "current_eye_closed_seconds": 0.0,
        "current_heavy_eye_seconds": 0.0,
        "score": None,
        "alert_level": None,
        "eye_status": "Unavailable",
        "mouth_status": "Unavailable",
        "head_status": "Unavailable",
        "fatigue_signs": "",
    }


def process_detected_face(frame, landmarks, analyzer, eye_state, row, timestamp_seconds):
    left_ear = calculate_ear(frame, landmarks, LEFT_EAR_POINTS)
    right_ear = calculate_ear(frame, landmarks, RIGHT_EAR_POINTS)
    ear = max(left_ear, right_ear)
    mar = calculate_mar(frame, landmarks, MOUTH_MAR_POINTS)
    head_pose = estimate_head_pose(frame, landmarks)

    eye_status, current_eye_closed_seconds, current_heavy_eye_seconds = eye_state.update(
        ear,
        timestamp_seconds,
    )
    mouth_status, _ = get_mouth_status(mar)
    head_status, _ = get_head_status(head_pose)
    pitch, yaw, roll = head_pose if head_pose is not None else (None, None, None)

    temporal_features = analyzer.add_sample(
        timestamp=timestamp_seconds,
        face_detected=True,
        ear=ear,
        raw_ear=ear,
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

    fatigue_sign_ids = merge_fatigue_signs(
        classify_instant_fatigue_signs(eye_status, mouth_status, head_status),
        classify_temporal_fatigue_signs(temporal_features),
    )
    scoring_features = {
        **temporal_features,
        "current_eye_closed_seconds": current_eye_closed_seconds,
        "current_heavy_eye_seconds": current_heavy_eye_seconds,
    }
    score, warning_signs = calculate_rule_based_score(
        eye_status,
        mouth_status,
        head_status,
        scoring_features,
        fatigue_sign_ids,
    )
    _, _, alert_level = get_alert_level(score)

    row.update(
        {
            "face_detected": True,
            "usable_for_training": True,
            "ear": ear,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "eye_closed_signal": 1.0 if eye_status != "Eyes open" else 0.0,
            "yawn_signal": 1.0 if mouth_status == "Yawning detected" else 0.0,
            "head_abnormal_signal": 1.0
            if any(sign_id in fatigue_sign_ids for sign_id in ABNORMAL_HEAD_SIGN_IDS)
            else 0.0,
            "current_eye_closed_seconds": current_eye_closed_seconds,
            "current_heavy_eye_seconds": current_heavy_eye_seconds,
            "score": score,
            "alert_level": alert_level,
            "eye_status": eye_status,
            "mouth_status": mouth_status,
            "head_status": head_status,
            "fatigue_signs": "|".join(format_sign_labels(fatigue_sign_ids)),
        }
    )

    return row, temporal_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build temporal feature CSV from labeled driver videos."
    )
    parser.add_argument("--video-root", default="dataset/videos")
    parser.add_argument("--output", default="data/features_video_temporal.csv")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Frames per second to process. Use 0 with --frame-step 1 for every frame.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Used only when --sample-fps is 0.",
    )
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    video_root = Path(args.video_root)
    output_path = Path(args.output)

    if not MODEL_PATH.exists():
        print(f"Missing model: {MODEL_PATH}", flush=True)
        sys.exit(1)

    if not video_root.exists():
        print(f"Missing video root: {video_root}", flush=True)
        sys.exit(1)

    if args.skip_existing and output_path.exists():
        print(f"Using existing CSV: {output_path}", flush=True)
        return

    videos = list_videos(video_root)
    if args.max_videos:
        videos = videos[: args.max_videos]

    if not videos:
        print(f"No videos found in {video_root}", flush=True)
        sys.exit(1)

    print(f"Found {len(videos)} videos in {video_root}", flush=True)
    print(f"Writing CSV to {output_path}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_faces = 0
    started_at = time.time()

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for video_path in videos:
            rows, faces = process_video(video_path, writer, args)
            total_rows += rows
            total_faces += faces
            csv_file.flush()

    elapsed = time.time() - started_at
    print(
        f"\nDone. Rows: {total_rows}, faces detected: {total_faces}, "
        f"elapsed: {format_duration(elapsed)}",
        flush=True,
    )
    print(f"CSV saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
