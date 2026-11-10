
from pathlib import Path

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

OUTPUT_DIR = PROJECT_ROOT / "eda_outputs"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "momtsim_v1_feature_target_relationships.csv"
)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print("MoMTSim V1 — Feature/Target Relationship Analysis")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Select candidate numerical features
# --------------------------------------------------

feature_columns = [
    "amount",
    "step",
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
    "initiator_balance_change",
    "recipient_balance_change",
    "initiator_balance_difference",
    "recipient_balance_difference",
]


# --------------------------------------------------
# 4. Analyze full dataset
# --------------------------------------------------

results = []

for feature in feature_columns:
    legitimate = df.loc[df["isFraud"] == 0, feature]
    fraud = df.loc[df["isFraud"] == 1, feature]

    results.append({
        "scope": "FULL_DATASET",
        "feature": feature,
        "legitimate_mean": legitimate.mean(),
        "fraud_mean": fraud.mean(),
        "legitimate_median": legitimate.median(),
        "fraud_median": fraud.median(),
        "mean_difference": fraud.mean() - legitimate.mean(),
        "median_difference": fraud.median() - legitimate.median(),
    })


# --------------------------------------------------
# 5. Analyze TRANSFER transactions only
# --------------------------------------------------

transfer_df = df.loc[
    df["transactionType"] == "TRANSFER"
]

for feature in feature_columns:
    legitimate = transfer_df.loc[
        transfer_df["isFraud"] == 0, feature
    ]

    fraud = transfer_df.loc[
        transfer_df["isFraud"] == 1, feature
    ]

    results.append({
        "scope": "TRANSFER_ONLY",
        "feature": feature,
        "legitimate_mean": legitimate.mean(),
        "fraud_mean": fraud.mean(),
        "legitimate_median": legitimate.median(),
        "fraud_median": fraud.median(),
        "mean_difference": fraud.mean() - legitimate.mean(),
        "median_difference": fraud.median() - legitimate.median(),
    })


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

for scope in ["FULL_DATASET", "TRANSFER_ONLY"]:

    print(f"\n{'=' * 65}")
    print(scope)
    print("=" * 65)

    scope_results = results_df.loc[
        results_df["scope"] == scope
    ]

    display_columns = [
        "feature",
        "legitimate_mean",
        "fraud_mean",
        "legitimate_median",
        "fraud_median",
        "mean_difference",
        "median_difference",
    ]

    print(
        scope_results[display_columns]
        .round(3)
        .to_string(index=False)
    )


# --------------------------------------------------
# 8. Completion
# --------------------------------------------------

print(f"\nResults saved to: {OUTPUT_FILE}")
print("\nFeature/target relationship analysis completed.")
