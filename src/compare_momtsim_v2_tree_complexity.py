import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

FEATURE = "oldBalInitiator"

train = pd.read_csv(
    "eda_outputs/momtsim_v2_train.csv",
    usecols=["transactionType", FEATURE, "isFraud"],
)

validation = pd.read_csv(
    "eda_outputs/momtsim_v2_validation.csv",
    usecols=["transactionType", FEATURE, "isFraud"],
)

train = train[train["transactionType"] == "TRANSFER"]
validation = validation[validation["transactionType"] == "TRANSFER"]

X_train = train[[FEATURE]]
y_train = train["isFraud"]

X_val = validation[[FEATURE]]
y_val = validation["isFraud"]

results = []

for depth in range(1, 9):

    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=500,
        random_state=42,
    )

    model.fit(X_train, y_train)

    train_scores = model.predict_proba(X_train)[:, 1]
    val_scores = model.predict_proba(X_val)[:, 1]

    predictions = model.predict(X_val)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    results.append({
        "depth": depth,
        "leaves": model.get_n_leaves(),
        "train_pr_auc": average_precision_score(y_train, train_scores),
        "validation_pr_auc": average_precision_score(y_val, val_scores),
        "validation_roc_auc": roc_auc_score(y_val, val_scores),
        "precision": precision_score(y_val, predictions),
        "recall": recall_score(y_val, predictions),
        "f1": f1_score(y_val, predictions),
        "false_positives": fp,
        "false_negatives": fn,
    })

results_df = pd.DataFrame(results)

print("MoMTSim V2 — Decision Tree Complexity Comparison")
print("=" * 110)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

results_df.to_csv(
    "eda_outputs/momtsim_v2_tree_complexity.csv",
    index=False,
)

print("\nSaved: eda_outputs/momtsim_v2_tree_complexity.csv")
