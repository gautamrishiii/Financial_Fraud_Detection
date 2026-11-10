
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
]

TOLERANCE = 0.02


def main():
    stats = defaultdict(lambda: {
        "total": 0,
        "fraud": 0,
        "legitimate": 0,
        "initiator_match": 0,
        "recipient_match": 0,
    })

    invalid_numeric = defaultdict(int)
    negative_values = defaultdict(int)
    zero_amounts = 0

    min_values = {column: float("inf") for column in NUMERIC_COLUMNS}
    max_values = {column: float("-inf") for column in NUMERIC_COLUMNS}

    total_rows = 0

    print("MoMTSim Version 1 — Numerical Analysis")
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

            values = {}

            for column in NUMERIC_COLUMNS:
                try:
                    value = float(row[column].strip())
                    values[column] = value

                    min_values[column] = min(
                        min_values[column], value
                    )
                    max_values[column] = max(
                        max_values[column], value
                    )

                    if value < 0:
                        negative_values[column] += 1

                except (ValueError, AttributeError):
                    invalid_numeric[column] += 1

            if len(values) != len(NUMERIC_COLUMNS):
                continue

            if values["amount"] == 0:
                zero_amounts += 1

            current = stats[transaction_type]
            current["total"] += 1

            if label == "1":
                current["fraud"] += 1
            elif label == "0":
                current["legitimate"] += 1

            initiator_change = (
                values["oldBalInitiator"]
                - values["newBalInitiator"]
            )

            recipient_change = (
                values["newBalRecipient"]
                - values["oldBalRecipient"]
            )

            if abs(initiator_change - values["amount"]) <= TOLERANCE:
                current["initiator_match"] += 1

            if abs(recipient_change - values["amount"]) <= TOLERANCE:
                current["recipient_match"] += 1

    print(f"\nTotal transactions: {total_rows:,}")

    print("\n1. Numerical column ranges")

    for column in NUMERIC_COLUMNS:
        print(
            f"{column}: "
            f"minimum={min_values[column]:,.2f}, "
            f"maximum={max_values[column]:,.2f}"
        )

    print("\n2. Numerical validation")

    print(f"Zero-amount transactions: {zero_amounts:,}")

    for column in NUMERIC_COLUMNS:
        print(
            f"{column}: "
            f"invalid={invalid_numeric[column]:,}, "
            f"negative={negative_values[column]:,}"
        )

    print("\n3. Transaction types and fraud distribution")

    for transaction_type in sorted(stats):
        current = stats[transaction_type]
        total = current["total"]

        fraud_rate = current["fraud"] / total * 100

        print(f"\nTransaction type: {transaction_type}")
        print(f"Total: {total:,}")
        print(
            f"Legitimate: {current['legitimate']:,}"
        )
        print(f"Fraud: {current['fraud']:,}")
        print(f"Fraud rate: {fraud_rate:.2f}%")

        print(
            "Initiator balance matches amount: "
            f"{current['initiator_match']:,} "
            f"({current['initiator_match'] / total * 100:.2f}%)"
        )

        print(
            "Recipient balance matches amount: "
            f"{current['recipient_match']:,} "
            f"({current['recipient_match'] / total * 100:.2f}%)"
        )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
