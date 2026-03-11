import os
import shutil

src_root = r"d:\WebStorm 2025.2.2\ЛАААААААААААААААБЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ\\3K\\2 C\\олип\\.venv\\downloads"
dst = r"d:\WebStorm 2025.2.2\ЛАААААААААААААААБЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫЫ\\3K\\2 C\\олип\\platform\\platform\\ml_contest\\tools\\downloads"

os.makedirs(dst, exist_ok=True)
for d in ("train", "test"):
    s = os.path.join(src_root, d)
    if os.path.exists(s):
        try:
            shutil.move(s, dst)
            print(f"Moved {d} to {dst}")
        except Exception as e:
            print(f"Failed to move {d}: {e}")
    else:
        print(f"Not found: {s}")
