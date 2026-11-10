
import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

amounts = {
    "0": [],
    "1": [],
}


def main():
    print("MoMTSim V1 — TRANSFER Amount Comparison")
    print("=" * 60)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row["transactionType"].strip() != "TRANSFER":
                continue

            label = row["isFraud"].strip()
            amount = float(row["amount"])

            if label in amounts:
                amounts[label].append(amount)

    for label, description in [
        ("0", "Legitimate TRANSFER"),
        ("1", "Fraud-labelled TRANSFER"),
    ]:
        values = np.array(amounts[label])

        print(f"\n{description}")
        print(f"Transactions: {len(values):,}")

        if len(values) == 0:
            continue

        print(f"Minimum: {np.min(values):,.2f}")
        print(f"Maximum: {np.max(values):,.2f}")
        print(f"Mean: {np.mean(values):,.2f}")
        print(f"Median: {np.median(values):,.2f}")

        for percentile in [75, 90, 95, 99]:
            value = np.percentile(values, percentile)

            print(
                f"{percentile}th percentile: "
                f"{value:,.2f}"
            )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
