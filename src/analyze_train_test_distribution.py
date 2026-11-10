
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


def main():
    groups = defaultdict(lambda: {
        "count": 0,
        "fraud": 0,
        "amount_sum": 0.0,
        "amount_min": float("inf"),
        "amount_max": float("-inf"),
        "amounts": [],
    })

    print("MoMTSim V1 — Training/Test Distribution Comparison")
    print("=" * 65)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            step = int(row["step"])
            amount = float(row["amount"])
            label = row["isFraud"].strip()
            transaction_type = row["transactionType"].strip()

            split = (
                "Training"
                if step <= TRAIN_END_STEP
                else "Testing"
            )

            current = groups[(split, transaction_type)]

            current["count"] += 1
            current["amount_sum"] += amount
            current["amount_min"] = min(
                current["amount_min"], amount
            )
            current["amount_max"] = max(
                current["amount_max"], amount
            )
            current["amounts"].append(amount)

            if label == "1":
                current["fraud"] += 1

    for split in ["Training", "Testing"]:
        print(f"\n{'=' * 20} {split} {'=' * 20}")

        for transaction_type in sorted(
            key[1] for key in groups if key[0] == split
        ):
            current = groups[(split, transaction_type)]
            count = current["count"]

            fraud_rate = current["fraud"] / count * 100
            mean_amount = current["amount_sum"] / count

            sorted_amounts = sorted(current["amounts"])

            median_amount = sorted_amounts[
                len(sorted_amounts) // 2
            ]

            print(f"\nTransaction type: {transaction_type}")
            print(f"Count: {count:,}")
            print(f"Fraud rate: {fraud_rate:.2f}%")
            print(f"Mean amount: {mean_amount:,.2f}")
            print(f"Median amount: {median_amount:,.2f}")
            print(f"Minimum amount: {current['amount_min']:,.2f}")
            print(f"Maximum amount: {current['amount_max']:,.2f}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
