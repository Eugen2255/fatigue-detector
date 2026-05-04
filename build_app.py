"""Build desktop app with PyInstaller."""
from pathlib import Path
import subprocess
import sys
import shutil

def main():
    root = Path(__file__).resolve().parent
    spec_file = root / "FatigueDetectorApp.spec"
    dist_dir = root / "dist"
    build_dir = root / "build"

    for d in [build_dir, dist_dir]:
        if d.exists():
            shutil.rmtree(d)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        str(spec_file), 
    ]

    print(f"Building: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(root))
    
    if result.returncode == 0:
        app_path = dist_dir / "FatigueDetectorApp"
        print(f"Output: {app_path.absolute()}")
    else:
        print("Build failed!")
    
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()