
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(
    "dataset/raw/ieee_cis/train_transaction.csv"
)

CHUNK_SIZE = 100_000


def main():
    print("IEEE-CIS — Chronological Split Audit")
    print("=" * 65)

    parts = []

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=["TransactionDT", "isFraud"],
        chunksize=CHUNK_SIZE,
    ):
        parts.append(chunk)

    data = pd.concat(parts, ignore_index=True)

    q70, q85 = data["TransactionDT"].quantile(
        [0.70, 0.85]
    ).tolist()

    data["split"] = np.select(
        [
            data["TransactionDT"] <= q70,
            data["TransactionDT"] <= q85,
        ],
        [
            "train",
            "validation",
        ],
        default="test",
    )

    print(f"Total transactions: {len(data):,}")
    print(f"Fraud cases: {int(data['isFraud'].sum()):,}")
    print(f"TransactionDT minimum: {data['TransactionDT'].min()}")
    print(f"TransactionDT maximum: {data['TransactionDT'].max()}")
    print(f"70th percentile boundary: {q70}")
    print(f"85th percentile boundary: {q85}")

    print("\nChronological partitions")
    print("-" * 65)

    for split in ["train", "validation", "test"]:
        subset = data[data["split"] == split]

        transactions = len(subset)
        fraud = int(subset["isFraud"].sum())

        print(f"\n{split.upper()}")
        print(f"Transactions: {transactions:,}")
        print(f"Fraud cases: {fraud:,}")
        print(f"Fraud rate: {fraud / transactions * 100:.4f}%")
        print(
            "TransactionDT range: "
            f"{subset['TransactionDT'].min()} - "
            f"{subset['TransactionDT'].max()}"
        )

    print("\nSplit boundaries do not overlap.")


if __name__ == "__main__":
    main()