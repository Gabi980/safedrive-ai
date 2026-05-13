import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from alert_system import calculate_drowsiness_score, get_alert_level
from config import (
    EYE_CLOSED_EAR_THRESHOLD,
    HEAVY_EYE_EAR_RATIO,
    MODEL_PATH,
    YAWN_MAR_THRESHOLD,
)
from facial_features import calculate_ear, calculate_mar, get_mouth_status
from head_pose import estimate_head_pose, get_head_status
from landmark_indexes import LEFT_EAR_POINTS, MOUTH_MAR_POINTS, RIGHT_EAR_POINTS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_LOG_EVERY = 100
FL3D_ANNOTATION_PATH = Path("classification_frames/annotations_all.json")
ABNORMAL_HEAD_STATUSES = {"Head down", "Looking sideways", "Head tilted"}
CSV_COLUMNS = [
    "source_dataset",
    "source_type",
    "image_path",
    "class_name",
    "scenario",
    "label",
    "label_name",
    "usable_for_training",
    "eye_closed_signal",
    "yawn_signal",
    "head_abnormal_signal",
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
    return sorted(
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalize_name(value):
    return value.lower().strip().replace("-", "_").replace(" ", "_")


def label_from_state(state):
    normalized = normalize_name(state)

    if normalized in {"alert", "open", "not_drowsy", "no_yawn", "normal"}:
        return "alert", 0, "alert"

    if normalized in {
        "drowsy",
        "close",
        "closed",
        "microsleep",
        "sleepy",
        "yawn",
        "yawning",
    }:
        return "drowsy", 1, "drowsy"

    raise ValueError(f"Unknown label/state: {state}")


def infer_standard_metadata(image_path, dataset_path):
    relative_parts = image_path.relative_to(dataset_path).parts
    state = relative_parts[0]
    class_name, label, label_name = label_from_state(state)
    scenario = relative_parts[1] if len(relative_parts) > 2 else normalize_name(state)

    return class_name, scenario, label, label_name


def calculate_average_ear(frame, landmarks):
    left_ear = calculate_ear(frame, landmarks, LEFT_EAR_POINTS)
    right_ear = calculate_ear(frame, landmarks, RIGHT_EAR_POINTS)
    return (left_ear + right_ear) / 2.0


def get_static_eye_status(ear):
    if ear >= EYE_CLOSED_EAR_THRESHOLD * HEAVY_EYE_EAR_RATIO:
        return "Eyes open"

    if ear >= EYE_CLOSED_EAR_THRESHOLD:
        return "Eyes heavy"

    return "Eyes closed"


def signal_from_status(status, positive_statuses):
    return 1.0 if status in positive_statuses else 0.0


def base_row(source_dataset, source_type, image_path, class_name, scenario, label, label_name):
    return {
        "source_dataset": source_dataset,
        "source_type": source_type,
        "image_path": str(image_path),
        "class_name": class_name,
        "scenario": scenario,
        "label": label,
        "label_name": label_name,
        "usable_for_training": False,
        "eye_closed_signal": None,
        "yawn_signal": None,
        "head_abnormal_signal": None,
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


def apply_annotation_hints(row):
    scenario = normalize_name(row["scenario"])

    if scenario == "microsleep":
        row["eye_closed_signal"] = max(row["eye_closed_signal"] or 0.0, 1.0)

    if scenario in {"yawn", "yawning"}:
        row["yawn_signal"] = max(row["yawn_signal"] or 0.0, 1.0)


def process_full_face_image(
    image_path,
    source_dataset,
    class_name,
    scenario,
    label,
    label_name,
    landmarker,
    source_type="full_face",
):
    row = base_row(
        source_dataset,
        source_type,
        image_path,
        class_name,
        scenario,
        label,
        label_name,
    )
    frame = cv2.imread(str(image_path))

    if frame is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return row

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

    row.update(
        {
            "usable_for_training": True,
            "eye_closed_signal": signal_from_status(eye_status, {"Eyes closed"}),
            "yawn_signal": signal_from_status(mouth_status, {"Yawning detected"}),
            "head_abnormal_signal": signal_from_status(head_status, ABNORMAL_HEAD_STATUSES),
            "ear": ear,
            "mar": mar,
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
    )

    if source_type == "full_face_annotation":
        apply_annotation_hints(row)

    return row


def crop_row(image_path, source_dataset, crop_type, state):
    class_name, label, label_name = label_from_state(state)
    row = base_row(
        source_dataset,
        crop_type,
        image_path,
        class_name,
        normalize_name(state),
        label,
        label_name,
    )

    if crop_type == "eye_crop":
        is_closed = normalize_name(state) in {"close", "closed"}
        row.update(
            {
                "usable_for_training": True,
                "eye_closed_signal": 1.0 if is_closed else 0.0,
                "yawn_signal": 0.0,
                "head_abnormal_signal": 0.0,
                "score": 25 if is_closed else 0,
                "alert_level": 0,
                "eye_status": "Eyes closed" if is_closed else "Eyes open",
                "mouth_status": "Unavailable",
                "head_status": "Unavailable",
            }
        )
    else:
        is_yawn = normalize_name(state) in {"yawn", "yawning"}
        row.update(
            {
                "usable_for_training": True,
                "eye_closed_signal": 0.0,
                "yawn_signal": 1.0 if is_yawn else 0.0,
                "head_abnormal_signal": 0.0,
                "score": 25 if is_yawn else 0,
                "alert_level": 0,
                "eye_status": "Unavailable",
                "mouth_status": "Yawning detected" if is_yawn else "Mouth normal",
                "head_status": "Unavailable",
            }
        )

    return row


def write_rows(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_rows(input_path):
    with input_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


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


def process_standard_dataset(
    dataset_path,
    output_path,
    landmarker,
    max_images=None,
    log_every=DEFAULT_LOG_EVERY,
):
    images = list_images(dataset_path)

    if max_images is not None:
        images = images[:max_images]

    rows = []
    detected = 0
    started_at = time.time()
    print(f"[{dataset_path.name}] standard folder dataset, found {len(images)} images", flush=True)

    for index, image_path in enumerate(images, start=1):
        try:
            class_name, scenario, label, label_name = infer_standard_metadata(
                image_path,
                dataset_path,
            )
            row = process_full_face_image(
                image_path,
                dataset_path.name,
                class_name,
                scenario,
                label,
                label_name,
                landmarker,
            )
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
    print_dataset_summary(dataset_path.name, rows, output_path, started_at)
    return rows


def process_annotation_dataset(
    dataset_path,
    output_path,
    landmarker,
    max_images=None,
    log_every=DEFAULT_LOG_EVERY,
):
    annotation_path = dataset_path / FL3D_ANNOTATION_PATH
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    items = sorted(annotations.items())

    if max_images is not None:
        items = items[:max_images]

    rows = []
    detected = 0
    started_at = time.time()
    print(f"[{dataset_path.name}] annotation dataset, found {len(items)} annotated images", flush=True)

    for index, (relative_image_path, metadata) in enumerate(items, start=1):
        try:
            clean_relative_path = relative_image_path.replace("\\", "/").lstrip("./")
            image_path = dataset_path / clean_relative_path
            driver_state = metadata["driver_state"]
            class_name, label, label_name = label_from_state(driver_state)
            row = process_full_face_image(
                image_path,
                dataset_path.name,
                class_name,
                normalize_name(driver_state),
                label,
                label_name,
                landmarker,
                source_type="full_face_annotation",
            )
        except Exception as error:
            print(f"[{dataset_path.name}] skipped {relative_image_path}: {error}", flush=True)
            row = None

        if row is not None:
            rows.append(row)
            if row["face_detected"]:
                detected += 1

        if index == 1 or index % log_every == 0 or index == len(items):
            print_progress(dataset_path.name, index, len(items), detected, started_at)

    write_rows(output_path, rows)
    print_dataset_summary(dataset_path.name, rows, output_path, started_at)
    return rows


def collect_crop_images(dataset_path):
    crop_items = []
    eyes_root = dataset_path / "eyes"
    yawn_root = dataset_path / "yawn"

    if eyes_root.exists():
        for image_path in list_images(eyes_root):
            state = image_path.parent.name
            crop_items.append((image_path, "eye_crop", state))

    if yawn_root.exists():
        for image_path in list_images(yawn_root):
            state = image_path.parent.name
            crop_items.append((image_path, "mouth_crop", state))

    return crop_items


def process_crop_dataset(
    dataset_path,
    output_path,
    max_images=None,
    log_every=DEFAULT_LOG_EVERY,
):
    crop_items = collect_crop_images(dataset_path)

    if max_images is not None:
        crop_items = crop_items[:max_images]

    rows = []
    started_at = time.time()
    print(f"[{dataset_path.name}] crop dataset, found {len(crop_items)} images", flush=True)

    for index, (image_path, crop_type, state) in enumerate(crop_items, start=1):
        try:
            row = crop_row(image_path, dataset_path.name, crop_type, state)
        except Exception as error:
            print(f"[{dataset_path.name}] skipped {image_path}: {error}", flush=True)
            row = None

        if row is not None:
            rows.append(row)

        if index == 1 or index % log_every == 0 or index == len(crop_items):
            print_progress(dataset_path.name, index, len(crop_items), 0, started_at)

    write_rows(output_path, rows)
    print_dataset_summary(dataset_path.name, rows, output_path, started_at)
    return rows


def print_dataset_summary(dataset_name, rows, output_path, started_at):
    elapsed = time.time() - started_at
    detected = sum(1 for row in rows if row.get("face_detected") is True)
    usable = sum(1 for row in rows if row.get("usable_for_training") is True)
    print(
        f"{dataset_name}: wrote {len(rows)} rows to {output_path} "
        f"({detected} faces detected, {usable} usable rows, {elapsed:.1f}s)",
        flush=True,
    )


def process_dataset(dataset_path, output_path, landmarker, max_images=None, log_every=DEFAULT_LOG_EVERY):
    if (dataset_path / FL3D_ANNOTATION_PATH).exists():
        return process_annotation_dataset(
            dataset_path,
            output_path,
            landmarker,
            max_images,
            log_every,
        )

    if (dataset_path / "eyes").exists() or (dataset_path / "yawn").exists():
        return process_crop_dataset(dataset_path, output_path, max_images, log_every)

    return process_standard_dataset(
        dataset_path,
        output_path,
        landmarker,
        max_images,
        log_every,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build feature CSV files from image datasets.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true")
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

    if not dataset_root.exists():
        print(f"Missing dataset root: {dataset_root}")
        sys.exit(1)

    datasets = [path for path in dataset_root.iterdir() if path.is_dir()]
    if args.datasets:
        selected_names = set(args.datasets)
        datasets = [path for path in datasets if path.name in selected_names]

    datasets = sorted(datasets, key=lambda path: path.name)

    if not datasets:
        print(f"No datasets found in {dataset_root}")
        sys.exit(1)

    landmarker = create_image_landmarker()
    combined_rows = []

    try:
        for dataset_path in datasets:
            output_path = output_dir / f"features_{dataset_path.name}.csv"

            if args.skip_existing and output_path.exists():
                print(f"[{dataset_path.name}] using existing {output_path}", flush=True)
                rows = read_rows(output_path)
            else:
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
        print(f"combined: wrote {len(combined_rows)} rows to {combined_output_path}", flush=True)
    finally:
        landmarker.close()


if __name__ == "__main__":
    main()
