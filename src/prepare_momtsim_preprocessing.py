
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

DATA_DIR = Path("eda_outputs")

TRAIN_FILE = DATA_DIR / "momtsim_v1_train.csv"
VALIDATION_FILE = DATA_DIR / "momtsim_v1_validation.csv"
TEST_FILE = DATA_DIR / "momtsim_v1_test.csv"


# --------------------------------------------------
# 2. Approved pre-transaction features
# --------------------------------------------------

NUMERIC_FEATURES = [
    "amount",
    "step",
    "oldBalInitiator",
    "oldBalRecipient",
]

CATEGORICAL_FEATURES = [
    "transactionType",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "isFraud"

EXCLUDED_FEATURES = [
    "newBalInitiator",
    "newBalRecipient",
    "initiator_balance_change",
    "recipient_balance_change",
    "initiator_balance_difference",
    "recipient_balance_difference",
    "initiator",
    "recipient",
    TARGET,
]


# --------------------------------------------------
# 3. Load the chronological splits
# --------------------------------------------------

print("Loading MoMTSim V1 chronological splits...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training rows:   {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")
print(f"Testing rows:    {len(test_df):,}")


# --------------------------------------------------
# 4. Validate the feature policy
# --------------------------------------------------

print("\nChecking the pre-transaction feature policy...")

assert len(FEATURES) == len(set(FEATURES)), (
    "Duplicate feature names detected."
)

assert not set(FEATURES).intersection(EXCLUDED_FEATURES), (
    "An excluded feature has entered the feature list."
)

for split_name, df in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Testing", test_df),
]:
    missing_columns = set(FEATURES + [TARGET]) - set(df.columns)

    assert not missing_columns, (
        f"{split_name} is missing columns: {missing_columns}"
    )

    unexpected_features = set(FEATURES).intersection(EXCLUDED_FEATURES)

    assert not unexpected_features, (
        f"Excluded features detected: {unexpected_features}"
    )

    assert df[TARGET].notna().all(), (
        f"{split_name} contains missing target values."
    )

    assert set(df[TARGET].unique()).issubset({0, 1}), (
        f"{split_name} contains unexpected target labels."
    )

print("Feature policy checks passed.")

print("\nApproved numerical features:")
for feature in NUMERIC_FEATURES:
    print(f"  - {feature}")

print("\nApproved categorical features:")
for feature in CATEGORICAL_FEATURES:
    print(f"  - {feature}")


# --------------------------------------------------
# 5. Separate features and target
# --------------------------------------------------

X_train = train_df[FEATURES].copy()
X_validation = validation_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = train_df[TARGET].copy()
y_validation = validation_df[TARGET].copy()
y_test = test_df[TARGET].copy()


# --------------------------------------------------
# 6. Build preprocessing pipelines
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)


# --------------------------------------------------
# 7. Fit on training data only
# --------------------------------------------------

print("\nFitting preprocessing on training data only...")

X_train_processed = preprocessor.fit_transform(X_train)

print("Transforming validation data...")
X_validation_processed = preprocessor.transform(X_validation)

print("Transforming test data...")
X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# 8. Validate transformed matrices
# --------------------------------------------------

for split_name, matrix in [
    ("Training", X_train_processed),
    ("Validation", X_validation_processed),
    ("Testing", X_test_processed),
]:
    assert matrix.shape[1] == len(
        preprocessor.get_feature_names_out()
    ), f"{split_name} feature count mismatch."

    assert np.isfinite(matrix.data if hasattr(matrix, "data") and hasattr(matrix, "tocsr") else matrix).all(), (
        f"{split_name} contains non-finite values."
    )

print("\nTransformed feature names:")
for feature in preprocessor.get_feature_names_out():
    print(f"  - {feature}")


# --------------------------------------------------
# 9. Verify target alignment
# --------------------------------------------------

assert len(X_train_processed) == len(y_train)
assert len(X_validation_processed) == len(y_validation)
assert len(X_test_processed) == len(y_test)

print("\nTarget alignment checks passed.")


# --------------------------------------------------
# 10. Final report
# --------------------------------------------------

print("\n" + "=" * 60)
print("MoMTSim V1 — Pre-Transaction Preprocessing Report")
print("=" * 60)

print(f"Training matrix:   {X_train_processed.shape}")
print(f"Validation matrix: {X_validation_processed.shape}")
print(f"Testing matrix:    {X_test_processed.shape}")

print("\nPreprocessing policy:")
print("- Numerical features: median imputation + standard scaling")
print("- Categorical features: most-frequent imputation + one-hot encoding")
print("- Preprocessor fitted on training data only")
print("- Validation and test data transformed without refitting")
print("- Post-transaction features excluded")

print("\nPreprocessing completed successfully.")
