
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "eda_outputs"
    / "momtsim_v1_engineered.csv"
)


# --------------------------------------------------
# 2. Load TRANSFER transactions
# --------------------------------------------------

print("MoMTSim V1 — Balance Mismatch Investigation")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

transfer_df = df.loc[
    df["transactionType"] == "TRANSFER"
].copy()


# --------------------------------------------------
# 3. Recalculate balance changes
# --------------------------------------------------

initiator_change = (
    transfer_df["oldBalInitiator"]
    - transfer_df["newBalInitiator"]
)

recipient_change = (
    transfer_df["newBalRecipient"]
    - transfer_df["oldBalRecipient"]
)


# --------------------------------------------------
# 4. Identify mismatches
# --------------------------------------------------

matches = np.isclose(
    initiator_change,
    recipient_change,
)

mismatches = transfer_df.loc[~matches].copy()

mismatches["initiator_change_check"] = (
    initiator_change.loc[~matches]
)

mismatches["recipient_change_check"] = (
    recipient_change.loc[~matches]
)

mismatches["absolute_difference"] = (
    mismatches["initiator_change_check"]
    - mismatches["recipient_change_check"]
).abs()


# --------------------------------------------------
# 5. Display mismatches
# --------------------------------------------------

print(f"Total TRANSFER transactions: {len(transfer_df):,}")
print(f"Mismatching transactions: {len(mismatches):,}")

display_columns = [
    "step",
    "amount",
    "isFraud",
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
    "initiator_change_check",
    "recipient_change_check",
    "absolute_difference",
]

if len(mismatches) > 0:
    print("\nMismatching transaction details:\n")

    print(
        mismatches[display_columns]
        .to_string(index=False)
    )

    print("\nMismatch summary:")
    print(
        mismatches["absolute_difference"]
        .describe()
        .to_string()
    )
else:
    print("\nNo mismatches found.")


print("\nInvestigation completed.")
