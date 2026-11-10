
import pandas as pd

DATA_PATH = "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"

df = pd.read_csv(DATA_PATH, dtype={
    "initiator": "string",
    "recipient": "string",
})

df["initiator_difference"] = (
    df["amount"] - (df["oldBalInitiator"] - df["newBalInitiator"])
).abs()

df["recipient_difference"] = (
    df["amount"] - (df["newBalRecipient"] - df["oldBalRecipient"])
).abs()

columns = [
    "step",
    "transactionType",
    "amount",
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
    "isFraud",
    "initiator_difference",
    "recipient_difference",
]

for transaction_type in sorted(df["transactionType"].unique()):
    subset = df[df["transactionType"] == transaction_type]

    print("\n" + "=" * 100)
    print(f"TRANSACTION TYPE: {transaction_type}")
    print("=" * 100)

    for label in [0, 1]:
        group = subset[subset["isFraud"] == label]

        if group.empty:
            continue

        print(f"\n{'FRAUD' if label == 1 else 'LEGITIMATE'} — matched examples")

        matched = group[
            (group["initiator_difference"] <= 0.01)
            & (group["recipient_difference"] <= 0.01)
        ]

        print(
            matched[columns].head(3).to_string(index=False)
            if not matched.empty else "No matched examples."
        )

        print(f"\n{'FRAUD' if label == 1 else 'LEGITIMATE'} — mismatched examples")

        mismatched = group[
            (group["initiator_difference"] > 0.01)
            | (group["recipient_difference"] > 0.01)
        ]

        print(
            mismatched[columns].head(3).to_string(index=False)
            if not mismatched.empty else "No mismatched examples."
        )

del df
