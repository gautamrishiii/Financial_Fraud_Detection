import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURES = [
    "amount",
    "oldBalInitiator",
    "oldBalRecipient",
]

train = pd.read_csv(
    "eda_outputs/momtsim_v2_train.csv",
    usecols=["transactionType", "step", "isFraud", *FEATURES],
)

validation = pd.read_csv(
    "eda_outputs/momtsim_v2_validation.csv",
    usecols=["transactionType", "step", "isFraud", *FEATURES],
)

train = train[train["transactionType"] == "TRANSFER"]
validation = validation[validation["transactionType"] == "TRANSFER"]

model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=500,
    random_state=42,
)

model.fit(train[FEATURES], train["isFraud"])

print("MoMTSim V2 — Decision Tree Temporal Stability")
print("=" * 85)

for start in range(128, 160, 8):
    end = start + 7

    window = validation[
        validation["step"].between(start, end)
    ]

    X = window[FEATURES]
    y = window["isFraud"]

    scores = model.predict_proba(X)[:, 1]

    print(f"\nSteps {start}-{end}")
    print(f"Transactions: {len(window):,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print(f"Fraud prevalence: {y.mean():.4%}")
    print(f"PR-AUC: {average_precision_score(y, scores):.4f}")
    print(f"PR baseline: {y.mean():.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, scores):.4f}")

print("\nTemporal stability analysis completed.")