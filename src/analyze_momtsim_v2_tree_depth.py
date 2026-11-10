import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURE = ["oldBalInitiator"]

train = pd.read_csv(
    "eda_outputs/momtsim_v2_train.csv",
    usecols=["transactionType", "oldBalInitiator", "isFraud"],
)

validation = pd.read_csv(
    "eda_outputs/momtsim_v2_validation.csv",
    usecols=["transactionType", "oldBalInitiator", "isFraud"],
)

train = train[train["transactionType"] == "TRANSFER"]
validation = validation[validation["transactionType"] == "TRANSFER"]

print("MoMTSim V2 — Initiator Balance Decision Tree Depth")
print("=" * 85)

for depth in [1, 2, 3, 4, 5, 6, 8]:

    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=500,
        random_state=42,
    )

    model.fit(train[FEATURE], train["isFraud"])

    scores = model.predict_proba(validation[FEATURE])[:, 1]

    pr_auc = average_precision_score(
        validation["isFraud"],
        scores,
    )

    roc_auc = roc_auc_score(
        validation["isFraud"],
        scores,
    )

    print(
        f"Depth: {depth:2d} | "
        f"Leaves: {model.get_n_leaves():3d} | "
        f"PR-AUC: {pr_auc:.4f} | "
        f"ROC-AUC: {roc_auc:.4f}"
    )

print("\nDepth comparison completed.")