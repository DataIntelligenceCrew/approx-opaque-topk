"""
Train an XGBoost model for used-car price prediction from Parquet files.

The directory must contain:
  - used_cars_train.parquet
  - used_cars_test.parquet
"""

import argparse
import math
from pathlib import Path
from typing import Tuple

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def load_split(
    train_path: Path,
    test_path: Path,
    label_col: str,
) -> Tuple[xgb.DMatrix, xgb.DMatrix, pd.Series, pd.Series]:
    """
    Load train/test Parquet files and construct DMatrix objects.

    :param train_path: Path to the training Parquet file.
    :param test_path: Path to the test Parquet file.
    :param label_col: Name of the target column.
    :return: (dtrain, dtest, y_train, y_test) 4-tuple. 
    """
    dtrain_df: pd.DataFrame = pd.read_parquet(train_path)
    dtest_df: pd.DataFrame = pd.read_parquet(test_path)

    # Safe-drop 'listing_id' if present; always drop label column
    drop_cols_train = [c for c in (label_col, "listing_id") if c in dtrain_df.columns]
    drop_cols_test = [c for c in (label_col, "listing_id") if c in dtest_df.columns]

    X_train = dtrain_df.drop(columns=drop_cols_train)
    y_train = dtrain_df[label_col]

    X_test = dtest_df.drop(columns=drop_cols_test)
    y_test = dtest_df[label_col]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    return dtrain, dtest, y_train, y_test


def train_xgboost_separate(
    train_file: Path,
    test_file: Path,
    label_col: str,
    random_state: int = 42,
) -> xgb.Booster:
    """
    Train an XGBoost regressor and report train/test RMSE.

    :param train_file: Path to training Parquet.
    :param test_file: Path to test Parquet.
    :param label_col: Target column name.
    :param random_state: Seed for reproducibility.
    :return: A trained XGBoost model.
    """
    dtrain, dtest, y_train, y_test = load_split(train_file, test_file, label_col)

    # Hyparparameters for model training
    params: dict[str, object] = {
        "objective": "reg:squarederror",
        "max_depth": 7,
        "eta": 0.3,
        "device": "cpu", 
        "subsample": 1.0,
        "sampling_method": "uniform",
        "lambda": 2.0,
        "alpha": 0.0,
        "tree_method": "hist",
        "num_parallel_tree": 1,
        "nthread": 2,
        "seed": random_state,
    }

    num_boost_round: int = 400
    evals = [(dtrain, "train"), (dtest, "test")]

    booster: xgb.Booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        verbose_eval=False,
    )

    # Evaluate RMSE on train/test
    y_train_pred = booster.predict(dtrain)
    y_test_pred = booster.predict(dtest)

    train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"[UsedCars Train] Train RMSE: {train_rmse:.6f}")
    print(f"[UsedCars Train] Test RMSE: {test_rmse:.6f}")

    return booster


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost on UsedCar Parquet splits and save model JSON.")
    parser.add_argument("data_dir", type=str, help="Directory containing used_cars_train.parquet and used_cars_test.parquet. Model will be saved here.")
    parser.add_argument("--label-col", type=str, default="price", help="Target column name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()

    train_path = data_dir / "used_cars_train.parquet"
    test_path = data_dir / "used_cars_test.parquet"
    out_path = data_dir / "usedcars_model.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    print(f"[UsedCars Train] Loading data from: {data_dir}")
    print(f"[UsedCars Train] Train: {train_path.name}")
    print(f"[UsedCars Train] Test: {test_path.name}")
    print(f"[UsedCars Train] Training XGBoost (seed={args.seed}) ...")

    booster = train_xgboost_separate(
        train_file=train_path,
        test_file=test_path,
        label_col=args.label_col,
        random_state=args.seed,
    )

    booster.save_model(str(out_path))
    print(f"[UsedCars Train] Saved model to: {out_path}")


if __name__ == "__main__":
    main()
