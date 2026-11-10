
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)


def main():
    initiators = set()
    recipients = set()

    initiator_lengths = set()
    recipient_lengths = set()

    print("MoMTSim V1 — Account Identifier Structure")
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

            initiators.add(initiator)
            recipients.add(recipient)

            initiator_lengths.add(len(initiator))
            recipient_lengths.add(len(recipient))

    shared = initiators.intersection(recipients)

    print("\nInitiator identifiers")
    print("-" * 45)
    print(f"Unique identifiers: {len(initiators):,}")
    print(f"Identifier lengths: {sorted(initiator_lengths)}")
    print(f"Examples: {sorted(initiators)[:10]}")

    print("\nRecipient identifiers")
    print("-" * 45)
    print(f"Unique identifiers: {len(recipients):,}")
    print(f"Identifier lengths: {sorted(recipient_lengths)}")
    print(f"Examples: {sorted(recipients)[:10]}")

    print("\nCross-role overlap")
    print("-" * 45)
    print(f"Identifiers appearing in both roles: {len(shared):,}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
