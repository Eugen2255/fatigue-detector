# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    # Fallback: текущая директория (если spec запускается из build_app.py)
    ROOT = Path.cwd()


MODELS_SRC = ROOT / "models"
MODELS_DST = "models"

EXCLUDE_MODULES = [
    "sympy", "pandas.tests", "numpy.tests",
    "PyQt5", "PyQt6", "tkinter", "wx", "kivy",
    "pytest", "test", "doctest", "pdb",
    "flask", "django", "requests", "urllib3", "httpx",
    "IPython", "jupyter", "notebook", "setuptools",
]

datas = [(str(MODELS_SRC), MODELS_DST)]
binaries = []
hiddenimports = [
    "cv2", "cv2.gapi_onnx", "cv2.gapi_fluid",
    "onnxruntime", "onnxruntime.capi",
    "mediapipe", "mediapipe.tasks", "mediapipe.tasks.vision",
    "joblib", "sklearn", "sklearn.ensemble",
]

tmp_ret = collect_all('mediapipe')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

a = Analysis(
    [str(ROOT / "src" / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT)],  # Подключаем папку с hook-pyside6.py
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDE_MODULES,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FatigueDetectorApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='FatigueDetectorApp',
)