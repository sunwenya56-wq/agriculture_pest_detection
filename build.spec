# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：onedir 交付，模型与配置随包携带。"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

PROJECT_ROOT = Path(SPECPATH)  # noqa: F821

datas = [
    (str(PROJECT_ROOT / "models" / "best.pt"), "models"),
    (str(PROJECT_ROOT / "models" / "models_config.json"), "models"),
]

hiddenimports = []
for package in ("ultralytics", "cv2"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    hiddenimports += pkg_hidden

a = Analysis(  # noqa: F821
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib.backends._backend_tk"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="智慧农业病虫害识别系统",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AgriPestDetection",
)
