
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(
    "dataset/raw/paysim/PS_20174392719_1491204439457_log.csv"
)

CHUNK_SIZE = 200_000


def main():
    print("PaySim — Raw Balance Change Audit")
    print("=" * 75)

    columns = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]

    summaries = []

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=columns,
        chunksize=CHUNK_SIZE,
    ):
        chunk["origin_delta"] = (
            chunk["newbalanceOrig"] - chunk["oldbalanceOrg"]
        )

        chunk["destination_delta"] = (
            chunk["newbalanceDest"] - chunk["oldbalanceDest"]
        )

        for account in ["origin", "destination"]:
            delta = chunk[f"{account}_delta"]

            chunk[f"{account}_unchanged"] = (
                np.abs(delta) <= 0.01
            )

            chunk[f"{account}_absolute_match"] = (
                np.abs(np.abs(delta) - chunk["amount"]) <= 0.01
            )

        grouped = chunk.groupby(["type", "isFraud"]).agg(
            transactions=("amount", "size"),
            origin_mean_delta=("origin_delta", "mean"),
            destination_mean_delta=("destination_delta", "mean"),
            origin_unchanged_pct=("origin_unchanged", "mean"),
            destination_unchanged_pct=("destination_unchanged", "mean"),
            origin_absolute_match_pct=("origin_absolute_match", "mean"),
            destination_absolute_match_pct=("destination_absolute_match", "mean"),
        ).reset_index()

        summaries.append(grouped)

    combined = pd.concat(summaries, ignore_index=True)

    result = (
        combined.groupby(["type", "isFraud"])
        .apply(
            lambda group: pd.Series({
                "transactions": group["transactions"].sum(),
                "origin_mean_delta": np.average(
                    group["origin_mean_delta"],
                    weights=group["transactions"],
                ),
                "destination_mean_delta": np.average(
                    group["destination_mean_delta"],
                    weights=group["transactions"],
                ),
                "origin_unchanged_pct": np.average(
                    group["origin_unchanged_pct"],
                    weights=group["transactions"],
                ) * 100,
                "destination_unchanged_pct": np.average(
                    group["destination_unchanged_pct"],
                    weights=group["transactions"],
                ) * 100,
                "origin_absolute_match_pct": np.average(
                    group["origin_absolute_match_pct"],
                    weights=group["transactions"],
                ) * 100,
                "destination_absolute_match_pct": np.average(
                    group["destination_absolute_match_pct"],
                    weights=group["transactions"],
                ) * 100,
            }),
            include_groups=False,
        )
        .reset_index()
    )

    print(result.to_string(
        index=False,
        float_format=lambda x: f"{x:,.3f}",
    ))


if __name__ == "__main__":
    main()