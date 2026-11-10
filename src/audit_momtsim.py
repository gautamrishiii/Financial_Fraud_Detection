
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)


def main():
    missing_counts = Counter()
    label_counts = Counter()

    total_rows = 0
    malformed_rows = 0

    print("MoMTSim Version 1 — Data Quality Audit")
    print("=" * 60)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.reader(file)

        header = next(reader)
        expected_columns = len(header)

        print(f"\nTotal columns: {expected_columns}")
        print("\nColumn names:")
        for column in header:
            print(f"- {column}")

        for row in reader:
            total_rows += 1

            if len(row) != expected_columns:
                malformed_rows += 1
                continue

            for index, value in enumerate(row):
                if not value.strip():
                    missing_counts[header[index]] += 1

            label = row[header.index("isFraud")].strip()
            label_counts[label] += 1

    print("\n1. Record integrity")
    print(f"Total data rows: {total_rows:,}")
    print(f"Expected columns per row: {expected_columns}")
    print(f"Malformed rows: {malformed_rows:,}")

    print("\n2. Missing or blank values")
    total_missing = sum(missing_counts.values())

    print(f"Total missing cells: {total_missing:,}")

    for column in header:
        count = missing_counts[column]
        percentage = (
            count / total_rows * 100 if total_rows else 0
        )
        print(
            f"{column}: {count:,} "
            f"({percentage:.2f}%)"
        )

    print("\n3. Fraud-label verification")
    for label, count in sorted(label_counts.items()):
        print(f"isFraud={label}: {count:,}")

    print("\nAudit completed.")


if __name__ == "__main__":
    main()
