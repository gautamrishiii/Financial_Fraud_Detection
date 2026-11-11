
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_DIR = Path("eda_outputs")
OUTPUT_DIR = Path("models/paysim_preprocessed")

CHUNK_SIZE = 200_000

SPLIT_ROWS = {
    "train": 5_787_030,
    "validation": 472_017,
    "test": 103_573,
}

NUMERIC_FEATURES = [
    "amount",
    "step",
    "oldbalanceOrg",
]

CATEGORIES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

FEATURE_NAMES = (
    [f"log_{name}" for name in NUMERIC_FEATURES]
    + [f"type_{name}" for name in CATEGORIES]
)


def transform_numeric(data):
    values = data[NUMERIC_FEATURES].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite numeric feature detected.")

    if (values < 0).any():
        raise ValueError("Negative value found before log transformation.")

    return np.log1p(values)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("PaySim — Feature Preprocessing")
    print("=" * 65)

    scaler = StandardScaler()

    # Fit scaler exclusively on training data.
    train_file = INPUT_DIR / "paysim_train.csv"

    for chunk in pd.read_csv(
        train_file,
        usecols=NUMERIC_FEATURES,
        chunksize=CHUNK_SIZE,
    ):
        scaler.partial_fit(transform_numeric(chunk))

    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
    print("Training-only scaler fitted and saved.")

    # Transform all three partitions using the training scaler.
    for split, expected_rows in SPLIT_ROWS.items():
        input_file = INPUT_DIR / f"paysim_{split}.csv"

        X_path = OUTPUT_DIR / f"X_{split}.npy"
        y_path = OUTPUT_DIR / f"y_{split}.npy"

        X = np.lib.format.open_memmap(
            X_path,
            mode="w+",
            dtype=np.float32,
            shape=(expected_rows, len(FEATURE_NAMES)),
        )

        y = np.lib.format.open_memmap(
            y_path,
            mode="w+",
            dtype=np.uint8,
            shape=(expected_rows,),
        )

        offset = 0

        for chunk in pd.read_csv(
            input_file,
            usecols=NUMERIC_FEATURES + ["type", "isFraud"],
            chunksize=CHUNK_SIZE,
        ):
            numeric = scaler.transform(
                transform_numeric(chunk)
            ).astype(np.float32)

            categorical = np.column_stack([
                (chunk["type"].to_numpy() == category)
                for category in CATEGORIES
            ]).astype(np.float32)

            features = np.column_stack([
                numeric,
                categorical,
            ])

            end = offset + len(chunk)

            if end > expected_rows:
                raise ValueError(f"{split}: unexpected extra rows.")

            X[offset:end] = features
            y[offset:end] = chunk["isFraud"].to_numpy(
                dtype=np.uint8
            )

            offset = end

        if offset != expected_rows:
            raise ValueError(
                f"{split}: expected {expected_rows:,} rows, "
                f"received {offset:,}."
            )

        X.flush()
        y.flush()

        print(f"\n{split.upper()}")
        print(f"Rows: {offset:,}")
        print(f"Features: {X.shape[1]}")
        print(f"Fraud cases: {int(y.sum()):,}")
        print(f"Saved: {X_path}")
        print(f"Saved: {y_path}")

        del X, y

    metadata = {
        "dataset": "PaySim",
        "features": FEATURE_NAMES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORIES,
        "target": "isFraud",
        "scaler_fit_on": "training_only",
        "transformation": "log1p_then_standard_scaling",
    }

    with open(OUTPUT_DIR / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nPaySim preprocessing completed successfully.")


if __name__ == "__main__":
    main()