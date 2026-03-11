"""
build_exe.py — Сборка .exe для AI Olympiad GUI.
Обходит проблему кириллицы в пути: копирует venv и проект во временную папку,
собирает .exe, копирует результат обратно.

Использование:
    python build_exe.py
"""
import os
import sys
import shutil
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FILES_TO_COPY = [
    'app.py',
    'runtime_hook_dlls.py',
    'utils',
    'models/best_model.pt',
    'results',
]


def main():
    tmp_root = os.path.join(tempfile.gettempdir(), 'ai_olympiad_build')
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root)
    print(f"[build] Temp build dir: {tmp_root}")

    for item in FILES_TO_COPY:
        src = os.path.join(SCRIPT_DIR, item)
        dst = os.path.join(tmp_root, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
            print(f"  Copied dir: {item}")
        elif os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  Copied file: {item}")
        else:
            print(f"  WARNING: {item} not found, skipping")

    venv_src = os.path.join(SCRIPT_DIR, '.venv')
    venv_dst = os.path.join(tmp_root, '.venv')
    print(f"[build] Copying venv to temp (this may take a minute)...")
    shutil.copytree(venv_src, venv_dst)
    print(f"  Copied .venv/")

    venv_python = os.path.join(venv_dst, 'Scripts', 'python.exe')
    if not os.path.isfile(venv_python):
        print(f"[build] ERROR: python.exe not found in copied venv")
        return 1
    print(f"[build] Python: {venv_python}")

    haar_src_path = None
    for root, dirs, files in os.walk(os.path.join(venv_dst, 'Lib', 'site-packages', 'cv2')):
        for f in files:
            if f == 'haarcascade_frontalface_default.xml':
                haar_src_path = os.path.join(root, f)
                break
        if haar_src_path:
            break

    haar_dst_dir = os.path.join(tmp_root, 'cv2_data')
    os.makedirs(haar_dst_dir, exist_ok=True)
    if haar_src_path:
        shutil.copy2(haar_src_path, haar_dst_dir)
        print(f"  Found haar cascade")
    else:
        print(f"  WARNING: haar cascade not found")

    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
import os
ROOT = {repr(tmp_root)}

a = Analysis(
    [os.path.join(ROOT, 'app.py')],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (os.path.join(ROOT, 'models', 'best_model.pt'), 'models'),
        (os.path.join(ROOT, 'results'), 'results'),
        (os.path.join(ROOT, 'utils'), 'utils'),
        (os.path.join(ROOT, 'cv2_data', 'haarcascade_frontalface_default.xml'), 'cv2/data'),
    ],
    hiddenimports=[
        'PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.sip',
        'cv2', 'torch', 'torchvision', 'torchvision.models',
        'numpy', 'PIL', 'json',
        'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends.backend_agg',
        'sklearn', 'sklearn.metrics', 'sklearn.preprocessing',
        'utils', 'utils.dataset', 'utils.metrics',
    ],
    hookspath=[],
    runtime_hooks=[os.path.join(ROOT, 'runtime_hook_dlls.py')],
    excludes=['tkinter', 'IPython', 'jupyter', 'notebook', 'pandas'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='AI_Olympiad_GUI',
    debug=False, strip=False, upx=True,
    console=False,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True,
    name='AI_Olympiad_GUI',
)
'''
    spec_path = os.path.join(tmp_root, 'build.spec')
    with open(spec_path, 'w', encoding='utf-8') as f:
        f.write(spec_content)
    print(f"[build] Spec written: {spec_path}")

    print(f"\n[build] Running PyInstaller (this may take 5-10 minutes)...\n")
    cmd = [
        venv_python, '-m', 'PyInstaller',
        spec_path,
        '--noconfirm',
        '--workpath', os.path.join(tmp_root, 'build'),
        '--distpath', os.path.join(tmp_root, 'dist'),
    ]
    result = subprocess.run(cmd, cwd=tmp_root)

    if result.returncode != 0:
        print(f"\n[build] ERROR: PyInstaller failed with code {result.returncode}")
        print(f"  Build dir preserved at: {tmp_root}")
        return 1

    dist_src = os.path.join(tmp_root, 'dist', 'AI_Olympiad_GUI')
    dist_dst = os.path.join(SCRIPT_DIR, 'dist', 'AI_Olympiad_GUI')

    if os.path.exists(dist_dst):
        shutil.rmtree(dist_dst)
    os.makedirs(os.path.dirname(dist_dst), exist_ok=True)
    shutil.copytree(dist_src, dist_dst)

    exe_path = os.path.join(dist_dst, 'AI_Olympiad_GUI.exe')
    print(f"\n{'='*60}")
    print(f"[build] SUCCESS!")
    print(f"  EXE: {exe_path}")
    print(f"  Folder: {dist_dst}")
    print(f"  Copy the 'AI_Olympiad_GUI' folder to the target PC")
    print(f"  and run AI_Olympiad_GUI.exe")
    print(f"{'='*60}")

    try:
        shutil.rmtree(tmp_root)
        print("[build] Temp dir cleaned up.")
    except Exception:
        print(f"[build] Note: temp dir remains at {tmp_root}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
