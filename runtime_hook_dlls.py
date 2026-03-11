# -*- coding: utf-8 -*-
"""
PyInstaller runtime hook — fix DLL loading on Cyrillic (non-ASCII) Windows paths.

When the .exe is placed in a folder with Cyrillic characters, torch cannot
load c10.dll because Windows DLL loader chokes on the non-ASCII path.
This hook converts the path to a short (8.3) Windows name (always ASCII)
and registers it as a DLL directory BEFORE torch is imported.
"""
import os
import sys
import ctypes


def _fix_dll_directories():
    if not getattr(sys, 'frozen', False):
        return

    base = sys._MEIPASS

    try:
        base.encode('ascii')
        return
    except UnicodeEncodeError:
        pass

    try:
        buf = ctypes.create_unicode_buffer(1024)
        length = ctypes.windll.kernel32.GetShortPathNameW(base, buf, 1024)
        if length > 0:
            short_base = buf.value
            for subdir in ['torch/lib', 'torch/bin', 'cv2', '.']:
                dll_dir = os.path.join(short_base, subdir.replace('/', os.sep))
                if os.path.isdir(dll_dir):
                    try:
                        os.add_dll_directory(dll_dir)
                    except (OSError, AttributeError):
                        pass
            os.environ['PATH'] = short_base + os.pathsep + os.environ.get('PATH', '')
            return
    except Exception:
        pass

    try:
        import shutil
        import tempfile
        tmp_dlls = os.path.join(tempfile.gettempdir(), '_ai_olympiad_dlls')
        torch_lib = os.path.join(base, 'torch', 'lib')
        if os.path.isdir(torch_lib) and not os.path.isdir(tmp_dlls):
            os.makedirs(tmp_dlls, exist_ok=True)
            for f in os.listdir(torch_lib):
                if f.endswith('.dll'):
                    src = os.path.join(torch_lib, f)
                    dst = os.path.join(tmp_dlls, f)
                    if not os.path.isfile(dst):
                        shutil.copy2(src, dst)
        if os.path.isdir(tmp_dlls):
            try:
                os.add_dll_directory(tmp_dlls)
            except (OSError, AttributeError):
                pass
            os.environ['PATH'] = tmp_dlls + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass


_fix_dll_directories()
