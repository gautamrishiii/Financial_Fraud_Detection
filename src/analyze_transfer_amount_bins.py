
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

# Amount intervals for TRANSFER transactions.
BINS = [
    (0, 10_000),
    (10_000, 20_000),
    (20_000, 30_000),
    (30_000, 50_000),
    (50_000, 100_000),
    (100_000, 200_000),
    (200_000, 500_000),
    (500_000, float("inf")),
]


def get_bin_label(lower, upper):
    if upper == float("inf"):
        return f"{lower:,.0f}+"

    return f"{lower:,.0f}–{upper:,.0f}"


def main():
    counts = defaultdict(lambda: {
        "legitimate": 0,
        "fraud": 0,
    })

    total_transfers = 0

    print("MoMTSim V1 — Fraud Rates by Transfer Amount")
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

            amount = float(row["amount"])
            label = row["isFraud"].strip()

            for lower, upper in BINS:
                if lower <= amount < upper:
                    if label == "0":
                        counts[lower]["legitimate"] += 1
                    elif label == "1":
                        counts[lower]["fraud"] += 1

                    total_transfers += 1
                    break

    print(f"\nTotal TRANSFER transactions: {total_transfers:,}")

    print(
        f"\n{'Amount range':<22}"
        f"{'Total':>12}"
        f"{'Fraud':>12}"
        f"{'Fraud rate':>14}"
    )

    print("-" * 60)

    for lower, upper in BINS:
        legitimate = counts[lower]["legitimate"]
        fraud = counts[lower]["fraud"]
        total = legitimate + fraud

        rate = fraud / total * 100 if total else 0

        print(
            f"{get_bin_label(lower, upper):<22}"
            f"{total:>12,}"
            f"{fraud:>12,}"
            f"{rate:>13.2f}%"
        )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
