import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

TRAIN_PATH = "eda_outputs/momtsim_v2_train.csv"
VALIDATION_PATH = "eda_outputs/momtsim_v2_validation.csv"

FEATURES = [
    "amount",
    "oldBalInitiator",
    "oldBalRecipient",
]

def load_transfers(path):
    df = pd.read_csv(
        path,
        usecols=["transactionType", "isFraud", *FEATURES],
    )

    df = df[df["transactionType"] == "TRANSFER"]

    return df[FEATURES], df["isFraud"]


X_train, y_train = load_transfers(TRAIN_PATH)
X_validation, y_validation = load_transfers(VALIDATION_PATH)

model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=500,
    random_state=42,
)

model.fit(X_train, y_train)

train_scores = model.predict_proba(X_train)[:, 1]
validation_scores = model.predict_proba(X_validation)[:, 1]

validation_predictions = model.predict(X_validation)

print("MoMTSim V2 — Shallow Decision Tree Analysis")
print("=" * 75)

print("\nLearned decision rules:")
print(export_text(model, feature_names=FEATURES, decimals=2))

print("\nTraining performance:")
print(f"PR-AUC:  {average_precision_score(y_train, train_scores):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_train, train_scores):.4f}")

print("\nValidation performance:")
print(
    f"PR-AUC:  "
    f"{average_precision_score(y_validation, validation_scores):.4f}"
)
print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_validation, validation_scores):.4f}"
)

print("\nValidation confusion matrix:")
print(confusion_matrix(y_validation, validation_predictions))

print("\nValidation classification report:")
print(
    classification_report(
        y_validation,
        validation_predictions,
        digits=4,
        zero_division=0,
    )
)

print("\nFeature importance:")
for feature, importance in zip(FEATURES, model.feature_importances_):
    print(f"  {feature}: {importance:.4f}")