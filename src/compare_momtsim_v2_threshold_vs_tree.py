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

def load_transfers(path):
    df = pd.read_csv(
        path,
        usecols=["transactionType", FEATURE, "isFraud"],
    )
    return df[df["transactionType"] == "TRANSFER"].copy()


train = load_transfers("eda_outputs/momtsim_v2_train.csv")
validation = load_transfers("eda_outputs/momtsim_v2_validation.csv")

X_train = train[[FEATURE]]
y_train = train["isFraud"]

X_val = validation[[FEATURE]]
y_val = validation["isFraud"]

# Baseline: predict fraud when balance is below the training median.
threshold = X_train[FEATURE].median()
threshold_predictions = (X_val[FEATURE] <= threshold).astype(int)

# Depth-6 decision tree.
tree = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=500,
    random_state=42,
)

tree.fit(X_train, y_train)

tree_scores = tree.predict_proba(X_val)[:, 1]
tree_predictions = tree.predict(X_val)

models = {
    "Median threshold": (
        threshold_predictions,
        threshold_predictions.astype(float),
    ),
    "Depth-6 decision tree": (
        tree_predictions,
        tree_scores,
    ),
}

print("MoMTSim V2 — Threshold vs Decision Tree")
print("=" * 75)
print(f"Training median balance threshold: {threshold:,.2f}")

for name, (predictions, scores) in models.items():

    print(f"\n{name}")
    print("-" * 50)

    print(f"PR-AUC:    {average_precision_score(y_val, scores):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_val, scores):.4f}")
    print(f"Precision: {precision_score(y_val, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_val, predictions, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_val, predictions, zero_division=0):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions))

print("\nComparison completed.")