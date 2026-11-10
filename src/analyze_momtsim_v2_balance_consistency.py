
from collections import defaultdict

import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
CHUNK_SIZE = 250_000
TOLERANCE = 0.01

stats = defaultdict(lambda: {
    "count": 0,
    "mismatches": 0,
    "absolute_difference_sum": 0.0,
    "maximum_absolute_difference": 0.0,
})

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):

    chunk["initiator_balance_change"] = (
        chunk["oldBalInitiator"] - chunk["newBalInitiator"]
    )

    chunk["recipient_balance_change"] = (
        chunk["newBalRecipient"] - chunk["oldBalRecipient"]
    )

    differences = {
        "Initiator": (
            chunk["amount"] - chunk["initiator_balance_change"]
        ),
        "Recipient": (
            chunk["amount"] - chunk["recipient_balance_change"]
        ),
    }

    for side, difference in differences.items():
        absolute_difference = difference.abs()

        summary = pd.DataFrame({
            "transactionType": chunk["transactionType"],
            "isFraud": chunk["isFraud"],
            "absolute_difference": absolute_difference,
        })

        summary["is_mismatch"] = (
            summary["absolute_difference"] > TOLERANCE
        )

        grouped = summary.groupby(
            ["transactionType", "isFraud"]
        ).agg(
            count=("absolute_difference", "size"),
            mismatches=("is_mismatch", "sum"),
            difference_sum=("absolute_difference", "sum"),
            maximum_difference=("absolute_difference", "max"),
        )

        for (transaction_type, label), row in grouped.iterrows():
            key = (side, transaction_type, label)
            stats[key]["count"] += int(row["count"])
            stats[key]["mismatches"] += int(row["mismatches"])
            stats[key]["absolute_difference_sum"] += float(
                row["difference_sum"]
            )
            stats[key]["maximum_absolute_difference"] = max(
                stats[key]["maximum_absolute_difference"],
                float(row["maximum_difference"]),
            )

print("MoMTSim V2 — Balance Consistency Audit")
print("=" * 75)
print(f"Mismatch tolerance: {TOLERANCE:.2f}")

for side in ["Initiator", "Recipient"]:
    print(f"\n{side} balance:")
    print("-" * 75)

    for transaction_type in sorted(
        {key[1] for key in stats if key[0] == side}
    ):
        for label in [0, 1]:
            key = (side, transaction_type, label)

            if key not in stats:
                continue

            result = stats[key]
            count = result["count"]
            mismatches = result["mismatches"]
            mean_difference = (
                result["absolute_difference_sum"] / count
            )

            name = "Legitimate" if label == 0 else "Fraud"

            print(
                f"{transaction_type:12} {name:12} "
                f"rows={count:>10,} "
                f"mismatches={mismatches:>10,} "
                f"mean_abs_diff={mean_difference:>12.4f} "
                f"max_abs_diff={result['maximum_absolute_difference']:>15.4f}"
            )
