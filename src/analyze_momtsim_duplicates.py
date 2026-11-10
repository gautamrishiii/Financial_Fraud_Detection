
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
    counts = Counter()
    total_rows = 0

    print("MoMTSim V1 — Exact Duplicate Analysis")
    print("=" * 65)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.reader(file)

        header = next(reader)

        for row in reader:
            total_rows += 1
            counts[tuple(row)] += 1

    unique_rows = len(counts)

    duplicate_excess = total_rows - unique_rows

    repeated_groups = sum(
        1 for count in counts.values()
        if count > 1
    )

    print(f"Total transactions: {total_rows:,}")
    print(f"Unique transaction records: {unique_rows:,}")
    print(f"Repeated record groups: {repeated_groups:,}")
    print(f"Duplicate excess records: {duplicate_excess:,}")

    if duplicate_excess:
        print("\nMost frequently repeated records:")

        repeated = sorted(
            (
                (count, record)
                for record, count in counts.items()
                if count > 1
            ),
            reverse=True
        )[:5]

        for count, record in repeated:
            print(f"\nOccurrences: {count}")
            print(dict(zip(header, record)))

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
