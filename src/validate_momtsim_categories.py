
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

EXPECTED_TYPES = {
    "DEBIT",
    "DEPOSIT",
    "PAYMENT",
    "TRANSFER",
    "WITHDRAWAL",
}

EXPECTED_LABELS = {"0", "1"}


def main():
    transaction_types = Counter()
    labels = Counter()
    type_label_pairs = Counter()

    invalid_types = Counter()
    invalid_labels = Counter()

    total_rows = 0

    print("MoMTSim V1 — Category and Label Validation")
    print("=" * 65)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            total_rows += 1

            transaction_type = row["transactionType"].strip()
            label = row["isFraud"].strip()

            transaction_types[transaction_type] += 1
            labels[label] += 1

            type_label_pairs[
                (transaction_type, label)
            ] += 1

            if transaction_type not in EXPECTED_TYPES:
                invalid_types[transaction_type] += 1

            if label not in EXPECTED_LABELS:
                invalid_labels[label] += 1

    print(f"\nTotal records: {total_rows:,}")

    print("\nTransaction type values")
    print("-" * 45)

    for value, count in sorted(transaction_types.items()):
        print(f"{value}: {count:,}")

    print("\nFraud label values")
    print("-" * 45)

    for value, count in sorted(labels.items()):
        print(f"{value}: {count:,}")

    print("\nTransaction type and label combinations")
    print("-" * 45)

    for (transaction_type, label), count in sorted(
        type_label_pairs.items()
    ):
        print(
            f"{transaction_type:12} | "
            f"Label {label} | "
            f"{count:,}"
        )

    print("\nValidation results")
    print("-" * 45)
    print(f"Unexpected transaction types: {sum(invalid_types.values()):,}")
    print(f"Unexpected fraud labels: {sum(invalid_labels.values()):,}")

    if invalid_types:
        print(f"Unexpected types: {dict(invalid_types)}")

    if invalid_labels:
        print(f"Unexpected labels: {dict(invalid_labels)}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
