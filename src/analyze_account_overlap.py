
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "dataset/raw/momtsim/"
    / "synthetic_mobile_money_transaction_dataset.csv"
)

TRAIN_END_STEP = 116


def main():
    training_initiators = set()
    testing_initiators = set()

    training_recipients = set()
    testing_recipients = set()

    print("MoMTSim V1 — Account Identifier Overlap")
    print("=" * 65)

    with FILE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            step = int(row["step"])

            initiator = row["initiator"].strip()
            recipient = row["recipient"].strip()

            if step <= TRAIN_END_STEP:
                training_initiators.add(initiator)
                training_recipients.add(recipient)
            else:
                testing_initiators.add(initiator)
                testing_recipients.add(recipient)

    def report(name, training, testing):
        overlap = training.intersection(testing)
        total = training.union(testing)

        print(f"\n{name}")
        print("-" * 45)
        print(f"Unique training identifiers: {len(training):,}")
        print(f"Unique testing identifiers: {len(testing):,}")
        print(f"Identifiers shared across periods: {len(overlap):,}")
        print(f"Total unique identifiers: {len(total):,}")

        training_coverage = (
            len(overlap) / len(testing) * 100
            if testing else 0
        )

        print(
            f"Testing identifiers previously seen in training: "
            f"{training_coverage:.2f}%"
        )

    report(
        "Initiator identifiers",
        training_initiators,
        testing_initiators
    )

    report(
        "Recipient identifiers",
        training_recipients,
        testing_recipients
    )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
