
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

print("MoMTSim V1 — TRANSFER Feature Redundancy Check")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

transfer_df = df.loc[
    df["transactionType"] == "TRANSFER"
].copy()

print(f"TRANSFER transactions: {len(transfer_df):,}")


# --------------------------------------------------
# 3. Define feature pairs
# --------------------------------------------------

feature_pairs = [
    (
        "initiator_balance_change",
        "recipient_balance_change",
    ),
    (
        "initiator_balance_difference",
        "recipient_balance_difference",
    ),
]


# --------------------------------------------------
# 4. Compare each pair
# --------------------------------------------------

for feature_a, feature_b in feature_pairs:

    values_a = transfer_df[feature_a].to_numpy()
    values_b = transfer_df[feature_b].to_numpy()

    exact_matches = np.equal(values_a, values_b)

    differences = np.abs(values_a - values_b)

    print(f"\nFeature A: {feature_a}")
    print(f"Feature B: {feature_b}")

    print(
        "Exact matching rows:",
        f"{int(exact_matches.sum()):,}",
    )

    print(
        "Different rows:",
        f"{int((~exact_matches).sum()):,}",
    )

    print(
        "Maximum absolute difference:",
        f"{differences.max():.10f}",
    )

    print(
        "Mean absolute difference:",
        f"{differences.mean():.10f}",
    )

    print(
        "Correlation:",
        f"{transfer_df[feature_a].corr(transfer_df[feature_b]):.8f}",
    )


# --------------------------------------------------
# 5. Check balance-change identity
# --------------------------------------------------

initiator_change = (
    transfer_df["oldBalInitiator"]
    - transfer_df["newBalInitiator"]
)

recipient_change = (
    transfer_df["newBalRecipient"]
    - transfer_df["oldBalRecipient"]
)

print("\nBalance-change identity check:")

print(
    "Rows where initiator and recipient changes match:",
    f"{int(np.isclose(initiator_change, recipient_change).sum()):,}",
)


print("\nRedundancy check completed.")
