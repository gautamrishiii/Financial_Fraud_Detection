
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/paysim/"
    / "PS_20174392719_1491204439457_log.csv"
)


def main():
    fraud_counts = Counter()
    flag_counts = Counter()
    combinations = Counter()

    print("PaySim Existing Fraud-Flag Analysis")
    print("=" * 55)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {"isFraud", "isFlaggedFraud"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Required columns missing: {required}"
            )

        for row in reader:
            fraud = row["isFraud"].strip()
            flagged = row["isFlaggedFraud"].strip()

            fraud_counts[fraud] += 1
            flag_counts[flagged] += 1
            combinations[(fraud, flagged)] += 1

    print("\n1. Actual dataset fraud labels")
    for label in sorted(fraud_counts):
        print(f"isFraud={label}: {fraud_counts[label]:,}")

    print("\n2. Existing system flags")
    for label in sorted(flag_counts):
        print(
            f"isFlaggedFraud={label}: "
            f"{flag_counts[label]:,}"
        )

    print("\n3. Relationship between labels and flags")
    print("Rows = isFraud; Columns = isFlaggedFraud")

    print(f"\n{'':20} {'Flag=0':>15} {'Flag=1':>15}")

    for fraud in sorted(fraud_counts):
        count_0 = combinations[(fraud, "0")]
        count_1 = combinations[(fraud, "1")]

        print(
            f"Fraud={fraud:<13} "
            f"{count_0:>15,} "
            f"{count_1:>15,}"
        )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
