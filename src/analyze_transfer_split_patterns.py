
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

TRAIN_END_STEP = 116


def percentile(sorted_values, percentage):
    position = (len(sorted_values) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)

    fraction = position - lower

    return (
        sorted_values[lower] * (1 - fraction)
        + sorted_values[upper] * fraction
    )


def main():
    amounts = defaultdict(list)

    print("MoMTSim V1 — Transfer Amounts by Split and Label")
    print("=" * 65)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row["transactionType"].strip() != "TRANSFER":
                continue

            step = int(row["step"])
            label = row["isFraud"].strip()
            amount = float(row["amount"])

            split = (
                "Training"
                if step <= TRAIN_END_STEP
                else "Testing"
            )

            amounts[(split, label)].append(amount)

    for split in ["Training", "Testing"]:
        for label in ["0", "1"]:
            values = sorted(amounts[(split, label)])

            if not values:
                print(f"\nNo records: {split}, label {label}")
                continue

            print(f"\n{split} — {'Fraud' if label == '1' else 'Legitimate'}")
            print("-" * 45)
            print(f"Transactions: {len(values):,}")
            print(f"Minimum: {values[0]:,.2f}")
            print(f"Median: {percentile(values, 0.50):,.2f}")
            print(f"90th percentile: {percentile(values, 0.90):,.2f}")
            print(f"95th percentile: {percentile(values, 0.95):,.2f}")
            print(f"99th percentile: {percentile(values, 0.99):,.2f}")
            print(f"Maximum: {values[-1]:,.2f}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
