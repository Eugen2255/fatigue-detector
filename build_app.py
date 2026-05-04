"""Build desktop app with PyInstaller."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--windowed",
        "--name",
        "FatigueDetectorApp",
        "--collect-all",
        "mediapipe",
        "--add-data",
        f"{root / 'models'};models",
        str(root / "src" / "app.py"),
    ]
    raise SystemExit(subprocess.run(cmd, cwd=str(root)).returncode)


if __name__ == "__main__":
    main()
