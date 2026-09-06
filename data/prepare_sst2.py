"""Download SST-2 from HuggingFace and save combined train+validation split as CSV.

Usage:
    python data/prepare_sst2.py

Output:
    data/sst2/sst2.csv  — columns: sentence, label (0=negative, 1=positive)
"""

import os
import pandas as pd
from datasets import load_dataset as hf_load_dataset


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "sst2")
    os.makedirs(out_dir, exist_ok=True)

    print("Downloading SST-2 from HuggingFace (glue/sst2)...")
    ds = hf_load_dataset("glue", "sst2")

    # GLUE test split has no labels (-1), so combine train + validation only
    train_df = ds["train"].to_pandas()[["sentence", "label"]]
    val_df = ds["validation"].to_pandas()[["sentence", "label"]]
    combined = pd.concat([train_df, val_df], ignore_index=True)

    out_path = os.path.join(out_dir, "sst2.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} rows to {out_path}")
    print(f"Label distribution:\n{combined['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()