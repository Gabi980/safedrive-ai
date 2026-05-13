from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from qt_interface import run_qt_interface  # noqa: E402


def main():
    run_qt_interface()


if __name__ == "__main__":
    main()
