
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)


def main():
    initiators = defaultdict(lambda: [0, 0])
    recipients = defaultdict(lambda: [0, 0])

    total_transactions = 0
    total_fraud = 0

    print("MoMTSim V1 — Account and Fraud Association")
    print("=" * 65)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            initiator = row["initiator"].strip()
            recipient = row["recipient"].strip()
            fraud = int(row["isFraud"])

            total_transactions += 1
            total_fraud += fraud

            initiators[initiator][0] += 1
            initiators[initiator][1] += fraud

            recipients[recipient][0] += 1
            recipients[recipient][1] += fraud

    print(f"\nTotal transactions: {total_transactions:,}")
    print(f"Total fraud transactions: {total_fraud:,}")
    print(f"Unique initiators: {len(initiators):,}")
    print(f"Unique recipients: {len(recipients):,}")

    def summarize(name, accounts):
        active = [
            (identifier, values[0], values[1])
            for identifier, values in accounts.items()
            if values[0] > 0
        ]

        fraud_only = sum(
            1 for _, count, fraud in active
            if count == fraud
        )

        mixed = sum(
            1 for _, count, fraud in active
            if 0 < fraud < count
        )

        legitimate_only = sum(
            1 for _, count, fraud in active
            if fraud == 0
        )

        print(f"\n{name}")
        print("-" * 45)
        print(f"Accounts with only legitimate transactions: {legitimate_only:,}")
        print(f"Accounts with only fraud transactions: {fraud_only:,}")
        print(f"Accounts with both labels: {mixed:,}")

        print("\nTop 10 accounts by transaction count:")

        top = sorted(
            active,
            key=lambda item: item[1],
            reverse=True
        )[:10]

        for identifier, count, fraud in top:
            rate = fraud / count * 100

            print(
                f"ID: {identifier} | "
                f"Transactions: {count:,} | "
                f"Fraud: {fraud:,} | "
                f"Fraud rate: {rate:.2f}%"
            )

    summarize("Initiator analysis", initiators)
    summarize("Recipient analysis", recipients)

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
