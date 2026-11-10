
from pathlib import Path
import warnings

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("eda_outputs")

TRAIN_FILE = DATA_DIR / "momtsim_v1_train.csv"
VALIDATION_FILE = DATA_DIR / "momtsim_v1_validation.csv"

TARGET = "isFraud"

FEATURE_GROUPS = {
    "A - Request context": {
        "numeric": ["amount"],
        "categorical": ["transactionType"],
    },
    "B - Temporal context": {
        "numeric": ["amount", "step"],
        "categorical": ["transactionType"],
    },
    "C - Pre-transaction balances": {
        "numeric": [
            "amount",
            "oldBalInitiator",
            "oldBalRecipient",
        ],
        "categorical": ["transactionType"],
    },
    "D - Combined features": {
        "numeric": [
            "amount",
            "step",
            "oldBalInitiator",
            "oldBalRecipient",
        ],
        "categorical": ["transactionType"],
    },
}


# --------------------------------------------------
# 2. Load training and validation data
# --------------------------------------------------

print("Loading training and validation data...")

required_columns = sorted({
    TARGET,
    "amount",
    "step",
    "transactionType",
    "oldBalInitiator",
    "oldBalRecipient",
})

train_df = pd.read_csv(TRAIN_FILE, usecols=required_columns)
validation_df = pd.read_csv(VALIDATION_FILE, usecols=required_columns)

print(f"Training rows:   {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")

X_train_all = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_validation_all = validation_df.drop(columns=[TARGET])
y_validation = validation_df[TARGET]

print("\nTraining fraud rate: "
      f"{y_train.mean() * 100:.2f}%")

print("Validation fraud rate: "
      f"{y_validation.mean() * 100:.2f}%")


# --------------------------------------------------
# 3. Evaluation function
# --------------------------------------------------

def evaluate_model(model, X, y, model_name):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    f1 = f1_score(
        y, predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        y, probabilities
    )

    roc_auc = roc_auc_score(
        y, probabilities
    )

    matrix = confusion_matrix(y, predictions)

    print(f"\n{model_name}")
    print("-" * 55)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    return {
        "experiment": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


# --------------------------------------------------
# 4. Dummy classifier reference
# --------------------------------------------------

print("\n" + "=" * 65)
print("DUMMY CLASSIFIER REFERENCE")
print("=" * 65)

dummy = DummyClassifier(strategy="prior")
dummy.fit(
    X_train_all[["amount"]],
    y_train,
)

dummy_results = evaluate_model(
    dummy,
    X_validation_all[["amount"]],
    y_validation,
    "Dummy - Majority class reference",
)


# --------------------------------------------------
# 5. Train Logistic Regression experiments
# --------------------------------------------------

results = []

for experiment_name, feature_config in FEATURE_GROUPS.items():

    numeric_features = feature_config["numeric"]
    categorical_features = feature_config["categorical"]

    features = numeric_features + categorical_features

    print("\n" + "=" * 65)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 65)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Fitting model on training data...")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model.fit(
            X_train_all[features],
            y_train,
        )

    for warning in caught_warnings:
        print(f"WARNING: {warning.message}")

    print("Evaluating on validation data...")

    result = evaluate_model(
        model,
        X_validation_all[features],
        y_validation,
        experiment_name,
    )

    results.append(result)


# --------------------------------------------------
# 6. Compare validation results
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_file = DATA_DIR / "momtsim_v1_baseline_results.csv"

results_df.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("BASELINE EXPERIMENT COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print(f"\nResults saved to: {output_file}")

print("\nImportant:")
print("- These are validation results only.")
print("- The test set has not been evaluated.")
print("- The default classification threshold is 0.50.")
print("- No claim of real-world performance is implied.")
