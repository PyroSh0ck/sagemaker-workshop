"""
Fixes the relative paths in the MURA CSVs and copies them into the MURA/ working directory
so EfficientNetFineTune.py can read them directly.
"""
import pandas as pd, os

MURA_ROOT = "../data/mura/MURA-v1.1"
PATH_PREFIX = "../data/mura/"  # prepend so paths resolve from MURA/ directory

for fname in ("train_image_paths.csv", "valid_image_paths.csv"):
    src = os.path.join(MURA_ROOT, fname)
    if not os.path.exists(src):
        print(f"NOT FOUND: {src}")
        continue
    df = pd.read_csv(src, header=None, names=['path'])
    df['path'] = PATH_PREFIX + df['path']
    df.to_csv(fname, index=False, header=False)
    print(f"Written {fname} with {len(df)} paths")
    print(f"  Sample: {df['path'].iloc[0]}")
