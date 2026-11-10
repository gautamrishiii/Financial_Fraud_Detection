import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

TRAIN_PATH = "eda_outputs/momtsim_v2_train.csv"
VALIDATION_PATH = "eda_outputs/momtsim_v2_validation.csv"

FEATURE_GROUPS = {
    "A — Initiator balance only": ["oldBalInitiator"],
    "B — Amount only": ["amount"],
    "C — Amount + initiator balance": [
        "amount",
        "oldBalInitiator",
    ],
    "D — All three features": [
        "amount",
        "oldBalInitiator",
        "oldBalRecipient",
    ],
}


def load_transfers(path):
    df = pd.read_csv(
        path,
        usecols=[
            "transactionType",
            "amount",
            "oldBalInitiator",
            "oldBalRecipient",
            "isFraud",
        ],
    )

    return df[df["transactionType"] == "TRANSFER"].copy()


train = load_transfers(TRAIN_PATH)
validation = load_transfers(VALIDATION_PATH)

print("MoMTSim V2 — Decision Tree Feature Ablation")
print("=" * 80)

for group_name, features in FEATURE_GROUPS.items():

    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=500,
        random_state=42,
    )

    model.fit(train[features], train["isFraud"])

    scores = model.predict_proba(validation[features])[:, 1]

    pr_auc = average_precision_score(
        validation["isFraud"],
        scores,
    )

    roc_auc = roc_auc_score(
        validation["isFraud"],
        scores,
    )

    print(f"\n{group_name}")
    print(f"Features: {features}")
    print(f"Validation PR-AUC:  {pr_auc:.4f}")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")

print("\nFeature ablation completed.")