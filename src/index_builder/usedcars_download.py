"""
Download the Kaggle 'US Used Cars Dataset' into a target directory using kagglehub 0.2.9.
Saves the file as 'usedcars.csv'.
"""

import argparse
import shutil
from pathlib import Path
import kagglehub


def download_us_used_cars(target_dir: Path) -> Path:
    """
    Download and copy the dataset CSV into target_dir as usedcars.csv.
    Args:
        target_dir: Directory where usedcars.csv will be saved.
    Returns:
        Path to the saved CSV file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "usedcars.csv"
    dataset_dir = Path(kagglehub.dataset_download("ananaymital/us-used-cars-dataset"))
    source_file = dataset_dir / "used_cars_data.csv"
    if out_file.exists():
        out_file.unlink()
    shutil.copy2(source_file, out_file)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle US Used Cars Dataset (CSV) into a target directory.")
    parser.add_argument("target_dir", type=Path, help="Directory where the dataset CSV will be saved.")
    args = parser.parse_args()
    
    out_file = download_us_used_cars(args.target_dir)
    print(f"Dataset saved to: {out_file}")


if __name__ == "__main__":
    main()
