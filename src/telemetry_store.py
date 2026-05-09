import json
import os
from pathlib import Path
import time
from uuid import uuid4

from config import TELEMETRY_MAX_RECORDS, TELEMETRY_PATH


def reset_telemetry():
    path = Path(TELEMETRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_telemetry({"latest": None, "records": []})


def read_telemetry():
    path = Path(TELEMETRY_PATH)

    if not path.exists():
        return {"latest": None, "records": []}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"latest": None, "records": []}


def update_telemetry(record, should_sample):
    data = read_telemetry()
    records = data.get("records", [])

    if should_sample:
        records.append(record)
        records = records[-TELEMETRY_MAX_RECORDS:]

    write_telemetry({"latest": record, "records": records})


def write_telemetry(data):
    path = Path(TELEMETRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}.{os.getpid()}.{uuid4().hex}.tmp")

    temp_path.write_text(json.dumps(data), encoding="utf-8")

    for _ in range(10):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError:
            time.sleep(0.02)

    try:
        temp_path.unlink()
    except OSError:
        pass
