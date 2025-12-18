import ast
import os
import numpy as np
import pandas as pd
from .config import CFG

def _parse_target(x):
    """Parse '[0,1,0,...]' -> list length 8 of 0/1. Return None if invalid."""
    try:
        arr = ast.literal_eval(x) if isinstance(x, str) else x
        if not isinstance(arr, (list, tuple)) or len(arr) != CFG.NUM_CLASSES:
            return None
        arr = [0 if int(v) == 0 else 1 for v in arr]  # force 0/1
        return arr
    except Exception:
        return None

def load_and_clean_df():
    if not CFG.CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CFG.CSV_PATH}")

    df_raw = pd.read_csv(CFG.CSV_PATH)
    print(f"Rows raw: {len(df_raw)}")

    # 1) drop NaN target
    df = df_raw.dropna(subset=["target"]).copy()
    print(f"Rows after dropna(target): {len(df)}")

    # 2) parse + validate target
    parsed = df["target"].apply(_parse_target)
    bad = parsed.isna().sum()
    if bad:
        print(f"Dropping bad target rows: {bad}")
    df = df[parsed.notna()].copy()
    df["target_list"] = parsed[parsed.notna()].tolist()

    # 3) split into 8 columns
    for i, c in enumerate(CFG.CLASSES):
        df[c] = df["target_list"].apply(lambda a: int(a[i]))

    # 4) validate image filename column
    if "images" not in df.columns:
        raise ValueError("CSV must contain column named 'images' (filename).")

    # 5) filter missing image files
    def exists_image(fname):
        return (CFG.IMG_DIR / str(fname)).exists()

    mask = df["images"].apply(exists_image)
    missing = (~mask).sum()
    if missing:
        print(f"Filtering missing images: {missing}")
    df = df[mask].copy()

    # 6) sanity checks
    y = df[CFG.CLASSES].values
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Targets contain values other than 0/1 after cleaning.")

    print(f"Rows after filtering images: {len(df)}")
    label_counts = df[CFG.CLASSES].sum().astype(int)
    print("Label counts:\n", label_counts)

    all_zero = (df[CFG.CLASSES].sum(axis=1) == 0).sum()
    print("Samples with all-zero labels:", int(all_zero))

    return df
