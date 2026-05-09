from pathlib import Path
import subprocess
import sys
import time
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DASHBOARD_URL = "http://localhost:8501"

sys.path.insert(0, str(SRC_DIR))

from face_landmarks import main as run_camera_app


def start_dashboard():
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "dashboard.py",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    webbrowser.open(DASHBOARD_URL)
    return process


def main():
    dashboard_process = start_dashboard()

    try:
        run_camera_app()
    finally:
        dashboard_process.terminate()


if __name__ == "__main__":
    main()
