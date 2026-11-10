
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "MoMTSim Version 1": (
        "dataset/raw/momtsim/"
        "synthetic_mobile_money_transaction_dataset.csv"
    ),
    "MoMTSim Version 2": (
        "dataset/raw/momtsim/"
        "MoMTSim_20240722202413_1000_dataset.csv"
    ),
    "IEEE-CIS Training Data": (
        "dataset/raw/ieee_cis/"
        "train_transaction.csv"
    ),
    "PaySim": (
        "dataset/raw/paysim/"
        "PS_20174392719_1491204439457_log.csv"
    ),
}


def count_fraud_labels(name, relative_path):
    file_path = PROJECT_ROOT / relative_path
    counts = Counter()

    print(f"\nAnalyzing {name}...")
    print(f"File: {file_path.name}")

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        if "isFraud" not in (reader.fieldnames or []):
            print("ERROR: isFraud column not found.")
            return

        for row in reader:
            label = row["isFraud"].strip()
            counts[label] += 1

    total = sum(counts.values())

    print("\nResults:")
    print(f"Total transactions: {total:,}")

    for label, description in [
        ("0", "Legitimate"),
        ("1", "Fraud")
    ]:
        count = counts[label]
        percentage = count / total * 100 if total else 0

        print(
            f"{description}: "
            f"{count:,} ({percentage:.2f}%)"
        )

    unexpected = set(counts) - {"0", "1"}

    if unexpected:
        print(f"Unexpected labels: {unexpected}")


def main():
    print("=" * 60)
    print("FRAUD LABEL ANALYSIS — ALL SELECTED DATASETS")
    print("=" * 60)

    for name, relative_path in DATASETS.items():
        count_fraud_labels(name, relative_path)

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
