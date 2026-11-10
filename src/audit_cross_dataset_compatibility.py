
from pathlib import Path

import pandas as pd


DATASETS = {
    "MoMTSim V1": {
        "path": "dataset/raw/momtsim/synthetic_mobile_money_transaction_dataset.csv",
        "target": "isFraud",
    },
    "MoMTSim V2": {
        "path": "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv",
        "target": "isFraud",
    },
    "PaySim": {
        "path": "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv",
        "target": "isFraud",
    },
    "IEEE-CIS train": {
        "path": "dataset/raw/ieee_cis/train_transaction.csv",
        "target": "isFraud",
    },
    "IEEE-CIS test": {
        "path": "dataset/raw/ieee_cis/test_transaction.csv",
        "target": None,
    },
}

CHUNK_SIZE = 200_000


def audit_dataset(name, config):
    path = Path(config["path"])
    target = config["target"]

    print("\n" + "=" * 75)
    print(f"DATASET: {name}")
    print("=" * 75)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    total_rows = 0
    missing_counts = None
    target_counts = {}

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

        if missing_counts is None:
            missing_counts = pd.Series(0, index=chunk.columns, dtype="int64")

        missing_counts += chunk.isna().sum()

        if target and target in chunk.columns:
            counts = chunk[target].value_counts(dropna=False)
            for label, count in counts.items():
                target_counts[label] = target_counts.get(label, 0) + int(count)

    columns = list(missing_counts.index)

    print(f"File: {path}")
    print(f"Rows: {total_rows:,}")
    print(f"Columns: {len(columns)}")
    print(f"Target: {target if target else 'Not available'}")

    if target:
        print("\nTarget distribution:")
        for label, count in sorted(target_counts.items()):
            print(f"  {label}: {count:,} ({count / total_rows:.4%})")

    print("\nColumns with missing values:")
    missing = missing_counts[missing_counts > 0]

    if missing.empty:
        print("  None")
    else:
        for column, count in missing.sort_values(ascending=False).items():
            print(f"  {column}: {count:,} ({count / total_rows:.2%})")


def main():
    for name, config in DATASETS.items():
        audit_dataset(name, config)


if __name__ == "__main__":
    main()
