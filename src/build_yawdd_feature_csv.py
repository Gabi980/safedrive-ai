import argparse
import csv
from pathlib import Path
import re
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from alert_system import calculate_rule_based_score, get_alert_level
from build_video_feature_csv import CSV_COLUMNS, TEMPORAL_FEATURE_COLUMNS, normalize_value
from config import (
    EYE_CLOSED_EAR_THRESHOLD,
    LONG_YAWN_SECONDS_THRESHOLD,
    MICROSLEEP_SECONDS_THRESHOLD,
    MODEL_PATH,
    YAWN_MAR_THRESHOLD,
)
from facial_features import calculate_ear, calculate_mar
from fatigue_signs import (
    classify_instant_fatigue_signs,
    classify_temporal_fatigue_signs,
    format_sign_labels,
    merge_fatigue_signs,
)
from head_pose import estimate_head_pose, get_head_status
from landmark_indexes import LEFT_EAR_POINTS, MOUTH_MAR_POINTS, RIGHT_EAR_POINTS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_CLASS_NAMES = {
    0: "microsleep",
    1: "neutral",
    2: "yawning",
}
CLASS_PRIORITY = {
    "neutral": 0,
    "yawning": 1,
    "microsleep": 2,
}
OUTPUT_COLUMNS = [*CSV_COLUMNS, "sample_weight"]


def create_image_landmarker():
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def list_images(dataset_root):
    return sorted(
        path
        for split in ("train", "valid", "test")
        for path in (dataset_root / split / "images").glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for_image(image_path):
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def parse_yolo_labels(label_path):
    if not label_path.exists():
        return []

    labels = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        class_name = YOLO_CLASS_NAMES.get(class_id)
        if class_name is None:
            continue

        labels.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "bbox": tuple(float(value) for value in parts[1:5]),
            }
        )

    return labels


def choose_label(labels):
    if not labels:
        return None

    return max(labels, key=lambda item: CLASS_PRIORITY[item["class_name"]])


def image_sequence_id(image_path):
    stem = image_path.stem
    stem = stem.split("_jpg.rf.")[0]
    stem = stem.split("_png.rf.")[0]
    return re.sub(r"-\d+$", "", stem)


def frame_index_from_name(image_path):
    match = re.search(r"_mp4-(\d+)_", image_path.name)
    if match:
        return int(match.group(1))

    return 0


def crop_from_yolo_bbox(frame, bbox, margin=0.18):
    height, width = frame.shape[:2]
    center_x, center_y, box_width, box_height = bbox
    box_width *= 1.0 + margin
    box_height *= 1.0 + margin

    x1 = int((center_x - box_width / 2.0) * width)
    y1 = int((center_y - box_height / 2.0) * height)
    x2 = int((center_x + box_width / 2.0) * width)
    y2 = int((center_y + box_height / 2.0) * height)

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return frame[y1:y2, x1:x2]


def detect_landmarks(landmarker, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    )
    if result.face_landmarks:
        return result.face_landmarks[0], frame

    return None, frame


def detect_landmarks_with_fallback(landmarker, frame, bbox):
    landmarks, used_frame = detect_landmarks(landmarker, frame)
    if landmarks is not None or bbox is None:
        return landmarks, used_frame

    cropped_frame = crop_from_yolo_bbox(frame, bbox)
    return detect_landmarks(landmarker, cropped_frame)


def label_to_binary(class_name):
    if class_name == "neutral":
        return 0, "alert"

    return 1, "drowsy"


def label_statuses(class_name):
    if class_name == "microsleep":
        return "Drowsiness warning", "Mouth normal"

    if class_name == "yawning":
        return "Eyes open", "Yawning detected"

    return "Eyes open", "Mouth normal"


def pseudo_temporal_features(class_name):
    is_microsleep = class_name == "microsleep"
    is_yawning = class_name == "yawning"
    features = {column: 0.0 for column in TEMPORAL_FEATURE_COLUMNS}

    for window_seconds in (10, 30, 60):
        prefix = f"temporal_{window_seconds}s"
        features[f"{prefix}_face_visible_duration"] = float(window_seconds)
        features[f"{prefix}_blink_rate_per_minute"] = 0.0
        features[f"{prefix}_avg_blink_duration"] = 0.0
        features[f"{prefix}_longest_eye_closure"] = 0.0
        features[f"{prefix}_slow_blink_count"] = 0
        features[f"{prefix}_perclos_percent"] = 0.0
        features[f"{prefix}_closed_eye_ratio"] = 0.0
        features[f"{prefix}_heavy_eye_ratio"] = 0.0

    if is_microsleep:
        for window_seconds in (10, 30, 60):
            prefix = f"temporal_{window_seconds}s"
            features[f"{prefix}_perclos_percent"] = 100.0
            features[f"{prefix}_closed_eye_ratio"] = 1.0
            features[f"{prefix}_avg_blink_duration"] = MICROSLEEP_SECONDS_THRESHOLD
            features[f"{prefix}_longest_eye_closure"] = MICROSLEEP_SECONDS_THRESHOLD
            features[f"{prefix}_slow_blink_count"] = 1
        features["temporal_60s_blink_count"] = 1
        features["temporal_60s_microsleep_event_count"] = 1
        features["temporal_60s_microsleep_active"] = 1
        features["temporal_60s_microsleep_recent"] = 1

    features["yawn_count_60s"] = 1 if is_yawning else 0
    features["yawn_count_120s"] = 1 if is_yawning else 0
    features["avg_yawn_duration_60s"] = 1.0 if is_yawning else 0.0
    features["avg_yawn_duration_120s"] = 1.0 if is_yawning else 0.0
    features["max_yawn_duration"] = 1.0 if is_yawning else 0.0
    features["long_yawn_count"] = (
        1 if is_yawning and 1.0 >= LONG_YAWN_SECONDS_THRESHOLD else 0
    )

    return features


def build_base_row(image_path, label_info, args):
    class_id = label_info["class_id"]
    class_name = label_info["class_name"]
    label, label_name = label_to_binary(class_name)
    split_name = image_path.parents[1].name
    sequence_id = image_sequence_id(image_path)
    frame_index = frame_index_from_name(image_path)

    return {
        "source_dataset": "yawdd",
        "source_type": "image_frame",
        "video_id": f"yawdd_{sequence_id}",
        "video_path": str(image_path),
        "participant_id": split_name,
        "original_label": class_id,
        "original_label_name": class_name,
        "label": label,
        "label_name": label_name,
        "frame_index": frame_index,
        "timestamp_seconds": 0.0,
        "fps": 0.0,
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
        "head_abnormal_signal": 0.0,
        "current_eye_closed_seconds": (
            MICROSLEEP_SECONDS_THRESHOLD if class_name == "microsleep" else 0.0
        ),
        "current_heavy_eye_seconds": 0.0,
        "score": None,
        "alert_level": None,
        "eye_status": "Unavailable",
        "mouth_status": "Unavailable",
        "head_status": "Unavailable",
        "fatigue_signs": "",
        "sample_weight": args.sample_weight,
    }


def process_image(image_path, landmarker, writer, args):
    labels = parse_yolo_labels(label_path_for_image(image_path))
    label_info = choose_label(labels)
    if label_info is None:
        return False, False

    row = build_base_row(image_path, label_info, args)
    temporal_features = pseudo_temporal_features(label_info["class_name"])

    frame = cv2.imread(str(image_path))
    if frame is None:
        writer.writerow({**row, **temporal_features})
        return True, False

    landmarks, feature_frame = detect_landmarks_with_fallback(
        landmarker,
        frame,
        label_info.get("bbox"),
    )
    if landmarks is None:
        writer.writerow({**row, **temporal_features})
        return True, False

    left_ear = calculate_ear(feature_frame, landmarks, LEFT_EAR_POINTS)
    right_ear = calculate_ear(feature_frame, landmarks, RIGHT_EAR_POINTS)
    ear = max(left_ear, right_ear)
    mar = calculate_mar(feature_frame, landmarks, MOUTH_MAR_POINTS)
    head_pose = estimate_head_pose(feature_frame, landmarks)
    pitch, yaw, roll = head_pose if head_pose is not None else (None, None, None)
    head_status, _ = get_head_status(head_pose)
    eye_status, mouth_status = label_statuses(label_info["class_name"])

    fatigue_sign_ids = merge_fatigue_signs(
        classify_instant_fatigue_signs(eye_status, mouth_status, head_status),
        classify_temporal_fatigue_signs(temporal_features),
    )
    scoring_features = {
        **temporal_features,
        "current_eye_closed_seconds": row["current_eye_closed_seconds"],
        "current_heavy_eye_seconds": row["current_heavy_eye_seconds"],
    }
    score, _ = calculate_rule_based_score(
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
            "score": score,
            "alert_level": alert_level,
            "eye_status": eye_status,
            "mouth_status": mouth_status,
            "head_status": head_status,
            "fatigue_signs": "|".join(format_sign_labels(fatigue_sign_ids)),
        }
    )
    output_row = {**row}
    for column in TEMPORAL_FEATURE_COLUMNS:
        output_row[column] = normalize_value(temporal_features.get(column))
    writer.writerow(output_row)
    return True, True


def print_progress(processed, total, detected, started_at):
    elapsed = time.time() - started_at
    speed = processed / elapsed if elapsed > 0 else 0.0
    remaining = total - processed
    eta = remaining / speed if speed > 0 else 0.0
    print(
        f"{processed}/{total} ({processed / max(total, 1) * 100:.1f}%) | "
        f"faces: {detected} | speed: {speed:.1f} images/s | ETA: {eta / 60:.1f}m",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an auxiliary temporal-compatible CSV from YawDD images."
    )
    parser.add_argument("--dataset-root", default="dataset/yawdd")
    parser.add_argument("--output", default="data/features_yawdd_temporal_aux.csv")
    parser.add_argument("--sample-weight", type=float, default=0.20)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    images = list_images(dataset_root)
    if args.max_images > 0:
        images = images[:args.max_images]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"YawDD root: {dataset_root}", flush=True)
    print(f"Images: {len(images)}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Sample weight: {args.sample_weight}", flush=True)

    processed = 0
    labeled = 0
    detected = 0
    started_at = time.time()
    landmarker = create_image_landmarker()
    try:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for image_path in images:
                wrote_row, face_detected = process_image(
                    image_path,
                    landmarker,
                    writer,
                    args,
                )
                processed += 1
                if wrote_row:
                    labeled += 1
                if face_detected:
                    detected += 1

                if (
                    processed == 1
                    or processed % args.log_every == 0
                    or processed == len(images)
                ):
                    print_progress(processed, len(images), detected, started_at)
    finally:
        landmarker.close()

    print("\nDone.", flush=True)
    print(f"Processed images: {processed}", flush=True)
    print(f"Labeled rows:     {labeled}", flush=True)
    print(f"Faces detected:   {detected}", flush=True)


if __name__ == "__main__":
    main()
