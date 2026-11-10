
import csv
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

BALANCE_COLUMNS = [
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
]

TOLERANCE = 0.02


def main():
    label_counts = Counter()
    type_counts = defaultdict(Counter)
    negative_counts = defaultdict(Counter)
    mismatch_counts = defaultdict(Counter)

    total_rows = 0

    print("MoMTSim V1 — Fraud Pattern Comparison")
    print("=" * 65)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            total_rows += 1

            label = row["isFraud"].strip()
            transaction_type = row["transactionType"].strip()

            label_counts[label] += 1
            type_counts[label][transaction_type] += 1

            balances = {
                column: float(row[column])
                for column in BALANCE_COLUMNS
            }

            for column, value in balances.items():
                if value < 0:
                    negative_counts[label][column] += 1

            amount = float(row["amount"])

            initiator_change = (
                balances["oldBalInitiator"]
                - balances["newBalInitiator"]
            )

            recipient_change = (
                balances["newBalRecipient"]
                - balances["oldBalRecipient"]
            )

            if abs(initiator_change - amount) > TOLERANCE:
                mismatch_counts[label]["initiator"] += 1

            if abs(recipient_change - amount) > TOLERANCE:
                mismatch_counts[label]["recipient"] += 1

    print(f"\nTotal records analyzed: {total_rows:,}")

    print("\n1. Transaction types by fraud label")

    for label in sorted(label_counts):
        print(f"\nisFraud={label}")
        print(f"Total: {label_counts[label]:,}")

        for transaction_type, count in sorted(
            type_counts[label].items()
        ):
            percentage = count / label_counts[label] * 100

            print(
                f"  {transaction_type}: "
                f"{count:,} ({percentage:.2f}%)"
            )

    print("\n2. Negative balances by fraud label")

    for label in sorted(label_counts):
        print(f"\nisFraud={label}")

        for column in BALANCE_COLUMNS:
            count = negative_counts[label][column]
            percentage = count / label_counts[label] * 100

            print(
                f"  {column}: {count:,} "
                f"({percentage:.2f}%)"
            )

    print("\n3. Balance mismatches by fraud label")

    for label in sorted(label_counts):
        print(f"\nisFraud={label}")

        for account_type in ["initiator", "recipient"]:
            count = mismatch_counts[label][account_type]
            percentage = count / label_counts[label] * 100

            print(
                f"  {account_type}: {count:,} "
                f"({percentage:.2f}%)"
            )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
