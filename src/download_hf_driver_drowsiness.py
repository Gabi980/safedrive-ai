import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DEFAULT_REPO_ID = "n7i5x9/driver-drowsiness-dataset"
DEFAULT_OUTPUT_ROOT = "dataset/huggingface_driver_drowsiness"


def normalize_label(label_name):
    normalized = label_name.lower().replace("-", "_").replace(" ", "_")

    if normalized == "drowsy":
        return "drowsy"

    if normalized in {"not_drowsy", "non_drowsy", "alert", "awake"}:
        return "alert"

    raise ValueError(f"Unknown label: {label_name}")


def export_dataset(repo_id, output_root):
    output_root = Path(output_root)
    dataset = load_dataset(repo_id)
    counts = {"alert": 0, "drowsy": 0}

    print(f"Loaded {repo_id}")

    for split_name, split_data in dataset.items():
        print(f"Exporting {split_name}: {len(split_data)} images")

        for index, row in enumerate(tqdm(split_data, desc=split_name), start=1):
            label_name = split_data.features["label"].int2str(row["label"])
            class_name = normalize_label(label_name)
            output_dir = output_root / class_name
            output_dir.mkdir(parents=True, exist_ok=True)

            image_path = output_dir / f"{split_name}_{index:06d}_{class_name}.jpg"
            if image_path.exists():
                counts[class_name] += 1
                continue

            image = row["image"].convert("RGB")
            image.save(image_path, quality=95)
            counts[class_name] += 1

        print(f"{split_name} done")

    print("Export complete")
    print(f"Output: {output_root}")
    print(f"Alert images:  {counts['alert']}")
    print(f"Drowsy images: {counts['drowsy']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and export the HuggingFace driver drowsiness dataset."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    export_dataset(args.repo_id, args.output_root)


if __name__ == "__main__":
    main()
