import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# =========================
# Column configuration
# =========================

# Keep these lists short and explicit; add to them as needed.
BOOLEAN_COLS: List[str] = ["frame_damaged", "has_accidents", "is_new"]
NUMERIC_COLS: List[str] = ["daysonmarket", "height", "horsepower", "length", "mileage", "seller_rating"]
CATEGORICAL_COLS: List[str] = []  # add names if/when you have real categorical columns
SPECIAL_COLS: List[str] = ["price", "listing_id"]  # never normalize these

USEFUL_COLS: List[str] = BOOLEAN_COLS + NUMERIC_COLS + CATEGORICAL_COLS + SPECIAL_COLS


# =========================
# Helpers
# =========================

def clean_numeric_scalar(value: object) -> Union[float, np.floating]:
    """
    Convert messy cell content to float.
    - '--' -> NaN
    - Extract first numeric token from strings like '70.1 in' or '  -12 '
    - Otherwise use pandas to_numeric (coerce errors to NaN).
    """
    if isinstance(value, str) and value.strip() == "--":
        return np.nan
    if isinstance(value, str):
        m = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", value)
        if m:
            return float(m.group(0))
    return pd.to_numeric(value, errors="coerce")


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Vectorized wrapper over clean_numeric_scalar with dtype float64."""
    return s.map(clean_numeric_scalar).astype("float64")


def to_float01_from_boolish(s: pd.Series) -> pd.Series:
    """
    Map common booleanish encodings to {1.0, 0.0}, else NaN.
    Accepts: 'True'/'False' (any case), 1/0, '1'/'0', 'yes'/'no'.
    """
    true_set = {"true", "1", "yes", "y", "t"}
    false_set = {"false", "0", "no", "n", "f"}

    def _map(x: object) -> float:
        if isinstance(x, (int, np.integer)):
            return 1.0 if int(x) == 1 else (0.0 if int(x) == 0 else np.nan)
        if isinstance(x, float) and (x == 0.0 or x == 1.0):
            return float(x)
        if isinstance(x, str):
            v = x.strip().lower()
            if v in true_set:
                return 1.0
            if v in false_set:
                return 0.0
        return np.nan

    return s.map(_map).astype("float64")


def compute_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Compute per-column statistics for imputation.
    - numeric: mean
    - boolean (now 0/1 floats): mean (proportion True)
    - categorical: mode (first if multi-modal), or None if no mode
    """
    stats: Dict[str, Dict[str, float | str]] = {}

    for col in NUMERIC_COLS:
        if col in df.columns:
            stats[col] = {"mean": float(df[col].mean(skipna=True))}

    for col in BOOLEAN_COLS:
        if col in df.columns:
            stats[col] = {"mean": float(df[col].mean(skipna=True))}

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode_series = df[col].mode(dropna=True)
            stats[col] = {"mode": (None if mode_series.empty else str(mode_series.iloc[0]))}

    return stats


def impute_inplace(df: pd.DataFrame, stats: Dict[str, Dict[str, Union[float, str]]]) -> None:
    """Impute numeric/boolean with mean; categorical with mode (in place)."""
    for col in NUMERIC_COLS:
        if col in df.columns and "mean" in stats.get(col, {}):
            df[col] = df[col].fillna(stats[col]["mean"])

    for col in BOOLEAN_COLS:
        if col in df.columns and "mean" in stats.get(col, {}):
            df[col] = df[col].fillna(stats[col]["mean"])

    for col in CATEGORICAL_COLS:
        if col in df.columns and "mode" in stats.get(col, {}):
            mode_val = stats[col]["mode"]
            if mode_val is not None:
                df[col] = df[col].fillna(mode_val)


def minmax_normalize_inplace(df: pd.DataFrame, exclude: Iterable[str]) -> None:
    """
    Min-max normalize all columns except those in `exclude`.
    Skips non-numeric columns; handles constant columns safely.
    """
    excluded = set(exclude)
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min(skipna=True)
            col_max = df[col].max(skipna=True)
            denom = (col_max - col_min)
            if pd.isna(col_min) or pd.isna(col_max):
                # All-NaN column; leave as-is
                continue
            if denom == 0:
                # Constant column -> map to 0.0
                df[col] = 0.0
            else:
                df[col] = (df[col] - col_min) / denom


def shuffle_and_split(
    df: pd.DataFrame,
    *,
    val_n: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Shuffle full dataframe and split:
      - val: exactly `val_n` rows
      - remaining split 2:1 into train:test
    """
    if val_n < 0 or val_n > len(df):
        raise ValueError(f"val_n must be in [0, {len(df)}], got {val_n}")

    df_shuf = df.sample(frac=1.0, random_state=seed, ignore_index=True)
    val = df_shuf.iloc[:val_n]
    remaining = df_shuf.iloc[val_n:]

    train_size = int(len(remaining) * 2 / 3)
    train = remaining.iloc[:train_size]
    test = remaining.iloc[train_size:]

    return train, test, val


# =========================
# Pipeline
# =========================

def run_pipeline(data_dir: Path, *, val_n: int, seed: int) -> None:
    """
    Full in-memory pipeline:
      1) Load CSV
      2) Select useful columns & deduplicate columns
      3) Clean numerics/booleans; cast listing_id to string
      4) Compute stats and impute
      5) Min-max normalize (excluding SPECIAL_COLS)
      6) Save cleaned full dataset + shuffled splits to same directory
    """
    in_csv = data_dir / "usedcars.csv"
    out_clean = data_dir / "used_cars_clean.parquet"
    out_train = data_dir / "used_cars_train.parquet"
    out_test = data_dir / "used_cars_test.parquet"
    out_val = data_dir / "used_cars_val.parquet"

    print(f"[1/7] Reading raw CSV: {in_csv}")
    if not in_csv.exists():
        raise FileNotFoundError(f"Raw file not found: {in_csv}")

    # Load as strings to control coercion ourselves; subset columns after load (safer if CSV has surprises).
    df_raw = pd.read_csv(in_csv, dtype=str, low_memory=False)
    print(f"       Loaded shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

    # Retain useful columns that exist; warn if any are missing.
    missing = [c for c in USEFUL_COLS if c not in df_raw.columns]
    if missing:
        print(f"       [warn] Missing expected columns: {missing}")

    keep_cols = [c for c in USEFUL_COLS if c in df_raw.columns]
    df = df_raw[keep_cols].copy()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    print(f"[2/7] Retained columns: {len(df.columns)} -> {list(df.columns)}")

    # Clean numeric columns
    present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    for c in present_numeric:
        df[c] = clean_numeric_series(df[c])
    print(f"[3/7] Cleaned numeric columns: {present_numeric}")

    # Clean boolean columns to {0.0, 1.0, NaN}
    present_bool = [c for c in BOOLEAN_COLS if c in df.columns]
    for c in present_bool:
        df[c] = to_float01_from_boolish(df[c])
    print(f"[4/7] Parsed boolean columns: {present_bool}")

    # listing_id as string, if present
    if "listing_id" in df.columns:
        df["listing_id"] = df["listing_id"].astype("string")
        print(f"       Cast 'listing_id' to string dtype")

    # Basic NA summary before imputation
    na_counts = df.isna().sum()
    print(f"[5/7] NA counts before imputation (top 10):")
    print(na_counts.sort_values(ascending=False).head(10).to_string())

    # Compute stats & impute
    stats = compute_stats(df)
    impute_inplace(df, stats)
    print("       Imputation complete.")
    # Show a couple of stats
    for k in (NUMERIC_COLS[:2] + BOOLEAN_COLS[:2] + CATEGORICAL_COLS[:1]):
        if k in stats:
            print(f"       Stat[{k}]: {stats[k]}")

    # Normalize (exclude SPECIAL_COLS)
    minmax_normalize_inplace(df, exclude=SPECIAL_COLS + CATEGORICAL_COLS)
    print(f"[6/7] Min-max normalization complete (excluded: {SPECIAL_COLS + CATEGORICAL_COLS})")

    # Save cleaned full dataset
    df.to_parquet(out_clean, index=False)
    print(f"       Wrote cleaned dataset: {out_clean} ({len(df):,} rows)")

    # Shuffle & split
    print(f"[7/7] Shuffling and splitting (val_n={val_n}, seed={seed})")
    train_df, test_df, val_df = shuffle_and_split(df, val_n=val_n, seed=seed)
    print(f"       Split sizes -> train: {len(train_df):,}, test: {len(test_df):,}, val: {len(val_df):,}")

    # Save splits
    train_df.to_parquet(out_train, index=False)
    test_df.to_parquet(out_test, index=False)
    val_df.to_parquet(out_val, index=False)
    print(f"       Wrote splits:\n         - {out_train}\n         - {out_test}\n         - {out_val}")

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, impute, normalize, and split usedcars.csv entirely in-memory."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing usedcars.csv; outputs are written here as Parquet.",
    )
    parser.add_argument(
        "--val-n",
        type=int,
        default=100_000,
        help="Exact number of rows for the validation split (default: 100000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for shuffling (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.data_dir, val_n=args.val_n, seed=args.seed)


if __name__ == "__main__":
    main()
