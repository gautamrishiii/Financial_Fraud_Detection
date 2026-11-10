
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

# Group simulation steps into intervals of 12.
INTERVAL_SIZE = 12


def main():
    counts = defaultdict(lambda: {
        "legitimate": 0,
        "fraud": 0,
    })

    total_transfers = 0

    print("MoMTSim V1 — Fraud Patterns Over Simulation Steps")
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

            interval_start = (
                step // INTERVAL_SIZE
            ) * INTERVAL_SIZE

            interval_end = interval_start + INTERVAL_SIZE - 1

            if label == "0":
                counts[interval_start]["legitimate"] += 1
            elif label == "1":
                counts[interval_start]["fraud"] += 1

            total_transfers += 1

    print(f"\nTotal TRANSFER transactions: {total_transfers:,}")

    print(
        f"\n{'Step range':<18}"
        f"{'Total':>12}"
        f"{'Fraud':>12}"
        f"{'Fraud rate':>14}"
    )

    print("-" * 56)

    for start in sorted(counts):
        end = start + INTERVAL_SIZE - 1

        legitimate = counts[start]["legitimate"]
        fraud = counts[start]["fraud"]
        total = legitimate + fraud

        rate = fraud / total * 100 if total else 0

        print(
            f"{start:03d}–{end:03d}"
            f"{total:>12,}"
            f"{fraud:>12,}"
            f"{rate:>13.2f}%"
        )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
