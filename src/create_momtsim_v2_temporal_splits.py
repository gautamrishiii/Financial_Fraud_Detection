import pandas as pd
from pathlib import Path

DATA_PATH = Path(
    "dataset/raw/momtsim/MoMTSim_20240722202413_1000_dataset.csv"
)

OUTPUT_DIR = Path("eda_outputs")
CHUNK_SIZE = 250_000

SPLITS = {
    "train": (0, 127),
    "validation": (128, 159),
    "test": (160, 192),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_paths = {
    name: OUTPUT_DIR / f"momtsim_v2_{name}.csv"
    for name in SPLITS
}

# Remove previous split outputs so this run starts cleanly.
for path in output_paths.values():
    if path.exists():
        path.unlink()

counts = {
    name: {"rows": 0, "fraud": 0, "min_step": None, "max_step": None}
    for name in SPLITS
}

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):

    for name, (start_step, end_step) in SPLITS.items():

        selected = chunk[
            chunk["step"].between(start_step, end_step)
        ]

        if selected.empty:
            continue

        path = output_paths[name]

        selected.to_csv(
            path,
            mode="a",
            header=not path.exists(),
            index=False,
        )

        stats = counts[name]
        stats["rows"] += len(selected)
        stats["fraud"] += int(selected["isFraud"].sum())

        chunk_min = int(selected["step"].min())
        chunk_max = int(selected["step"].max())

        stats["min_step"] = (
            chunk_min if stats["min_step"] is None
            else min(stats["min_step"], chunk_min)
        )

        stats["max_step"] = (
            chunk_max if stats["max_step"] is None
            else max(stats["max_step"], chunk_max)
        )

print("MoMTSim V2 — Chronological Split Summary")
print("=" * 70)

for name, stats in counts.items():

    rows = stats["rows"]
    fraud = stats["fraud"]
    legitimate = rows - fraud

    print(f"\n{name.upper()}")
    print(f"  Rows:        {rows:,}")
    print(f"  Legitimate:  {legitimate:,}")
    print(f"  Fraud:       {fraud:,}")
    print(f"  Fraud rate:  {fraud / rows * 100:.2f}%")
    print(f"  Step range:  {stats['min_step']}–{stats['max_step']}")
    print(f"  File:        {output_paths[name]}")

total_rows = sum(s["rows"] for s in counts.values())
total_fraud = sum(s["fraud"] for s in counts.values())

print("\n" + "=" * 70)
print(f"Total split rows:  {total_rows:,}")
print(f"Total split fraud: {total_fraud:,}")

assert total_rows == 4_225_958
assert total_fraud == 2_233_118

print("\nAll row-count and fraud-count checks passed.")