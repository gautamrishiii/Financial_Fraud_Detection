import pandas as pd
from pathlib import Path

SPLITS = {
    "TRAIN": Path("eda_outputs/momtsim_v2_train.csv"),
    "VALIDATION": Path("eda_outputs/momtsim_v2_validation.csv"),
    "TEST": Path("eda_outputs/momtsim_v2_test.csv"),
}

COLUMNS = [
    "step",
    "amount",
    "oldBalInitiator",
    "oldBalRecipient",
    "isFraud",
]

print("MoMTSim V2 — Legitimate Low-Amount Transfer Exceptions")
print("=" * 80)

total_exceptions = 0

for split_name, path in SPLITS.items():

    df = pd.read_csv(path, usecols=COLUMNS + ["transactionType"])

    exceptions = df[
        (df["transactionType"] == "TRANSFER")
        & (df["amount"] < 1_000)
        & (df["isFraud"] == 0)
    ][COLUMNS]

    print(f"\n{split_name}: {len(exceptions)} exceptions")

    if not exceptions.empty:
        print(exceptions.to_string(index=False))

    total_exceptions += len(exceptions)
    del df

print("\n" + "=" * 80)
print(f"Total legitimate exceptions: {total_exceptions}")