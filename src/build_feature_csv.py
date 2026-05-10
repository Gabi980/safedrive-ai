import argparse
import csv
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from alert_system import calculate_drowsiness_score, get_alert_level
from config import EYE_CLOSED_EAR_THRESHOLD, MODEL_PATH, YAWN_MAR_THRESHOLD
from facial_features import calculate_ear, calculate_mar, get_mouth_status
from head_pose import estimate_head_pose, get_head_status
from landmark_indexes import LEFT_EAR_POINTS, MOUTH_MAR_POINTS, RIGHT_EAR_POINTS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_LOG_EVERY = 100
CSV_COLUMNS = [
    "source_dataset",
    "image_path",
    "class_name",
    "scenario",
    "label",
    "label_name",
    "ear",
    "ear_threshold",
    "mar",
    "mar_threshold",
    "pitch",
    "yaw",
    "roll",
    "score",
    "alert_level",
    "eye_status",
    "mouth_status",
    "head_status",
    "face_detected",
]


def create_image_landmarker():
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def list_images(dataset_path):
    return [
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def infer_metadata(image_path, dataset_path):
    relative_parts = image_path.relative_to(dataset_path).parts
    class_name = relative_parts[0].lower().replace(" ", "_")
    scenario = relative_parts[1] if len(relative_parts) > 2 else class_name

    if class_name == "alert":
        return class_name, scenario, 0, "alert"

    if class_name == "drowsy":
        return class_name, scenario, 1, "drowsy"

    raise ValueError(f"Unknown class folder: {class_name}")


def calculate_average_ear(frame, landmarks):
    left_ear = calculate_ear(frame, landmarks, LEFT_EAR_POINTS)
    right_ear = calculate_ear(frame, landmarks, RIGHT_EAR_POINTS)
    return (left_ear + right_ear) / 2.0


def get_static_eye_status(ear):
    if ear >= EYE_CLOSED_EAR_THRESHOLD:
        return "Eyes open"

    return "Eyes closed"


def process_image(image_path, dataset_path, source_dataset, landmarker):
    class_name, scenario, label, label_name = infer_metadata(image_path, dataset_path)
    frame = cv2.imread(str(image_path))

    if frame is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {
            "source_dataset": source_dataset,
            "image_path": str(image_path),
            "class_name": class_name,
            "scenario": scenario,
            "label": label,
            "label_name": label_name,
            "ear": None,
            "ear_threshold": EYE_CLOSED_EAR_THRESHOLD,
            "mar": None,
            "mar_threshold": YAWN_MAR_THRESHOLD,
            "pitch": None,
            "yaw": None,
            "roll": None,
            "score": None,
            "alert_level": None,
            "eye_status": "Unavailable",
            "mouth_status": "Unavailable",
            "head_status": "Unavailable",
            "face_detected": False,
        }

    landmarks = result.face_landmarks[0]
    ear = calculate_average_ear(frame, landmarks)
    mar = calculate_mar(frame, landmarks, MOUTH_MAR_POINTS)
    head_pose = estimate_head_pose(frame, landmarks)

    eye_status = get_static_eye_status(ear)
    mouth_status, _ = get_mouth_status(mar)
    head_status, _ = get_head_status(head_pose)

    score = calculate_drowsiness_score(eye_status, mouth_status, head_status)
    _, _, alert_level = get_alert_level(score)
    pitch, yaw, roll = head_pose if head_pose is not None else (None, None, None)

    return {
        "source_dataset": source_dataset,
        "image_path": str(image_path),
        "class_name": class_name,
        "scenario": scenario,
        "label": label,
        "label_name": label_name,
        "ear": ear,
        "ear_threshold": EYE_CLOSED_EAR_THRESHOLD,
        "mar": mar,
        "mar_threshold": YAWN_MAR_THRESHOLD,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "score": score,
        "alert_level": alert_level,
        "eye_status": eye_status,
        "mouth_status": mouth_status,
        "head_status": head_status,
        "face_detected": True,
    }


def write_rows(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


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


def print_progress(dataset_name, index, total, detected, started_at):
    elapsed = time.time() - started_at
    images_per_second = index / elapsed if elapsed > 0 else 0
    remaining = total - index
    eta = remaining / images_per_second if images_per_second > 0 else 0
    percent = (index / total) * 100 if total else 100

    print(
        f"[{dataset_name}] {index}/{total} ({percent:.1f}%) | "
        f"faces: {detected} | speed: {images_per_second:.1f} img/s | "
        f"elapsed: {format_duration(elapsed)} | ETA: {format_duration(eta)}",
        flush=True,
    )


def process_dataset(dataset_path, output_path, landmarker, max_images=None, log_every=DEFAULT_LOG_EVERY):
    images = list_images(dataset_path)

    if max_images is not None:
        images = images[:max_images]

    rows = []
    detected = 0
    started_at = time.time()
    print(f"[{dataset_path.name}] found {len(images)} images", flush=True)

    for index, image_path in enumerate(images, start=1):
        try:
            row = process_image(image_path, dataset_path, dataset_path.name, landmarker)
        except Exception as error:
            print(f"[{dataset_path.name}] skipped {image_path}: {error}", flush=True)
            row = None

        if row is not None:
            rows.append(row)
            if row["face_detected"]:
                detected += 1

        if index == 1 or index % log_every == 0 or index == len(images):
            print_progress(dataset_path.name, index, len(images), detected, started_at)

    write_rows(output_path, rows)
    elapsed = time.time() - started_at
    print(
        f"{dataset_path.name}: wrote {len(rows)} rows to {output_path} "
        f"({detected} faces detected, {elapsed:.1f}s)"
    )

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build feature CSV files from image datasets.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--max-images-per-dataset", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    if not MODEL_PATH.exists():
        print(f"Missing model: {MODEL_PATH}")
        sys.exit(1)

    datasets = [path for path in dataset_root.iterdir() if path.is_dir()]

    if not datasets:
        print(f"No datasets found in {dataset_root}")
        sys.exit(1)

    landmarker = create_image_landmarker()
    combined_rows = []

    try:
        for dataset_path in datasets:
            output_path = output_dir / f"features_{dataset_path.name}.csv"
            rows = process_dataset(
                dataset_path,
                output_path,
                landmarker,
                args.max_images_per_dataset,
                args.log_every,
            )
            combined_rows.extend(rows)

        combined_output_path = output_dir / "features_combined.csv"
        write_rows(combined_output_path, combined_rows)
        print(f"combined: wrote {len(combined_rows)} rows to {combined_output_path}")
    finally:
        landmarker.close()


if __name__ == "__main__":
    main()
