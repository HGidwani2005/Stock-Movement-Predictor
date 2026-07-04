import os
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

for file in files:
    print(f"Processing {file}...")

    path = os.path.join(RAW_DIR, file)
    df = pd.read_csv(path)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Remove missing values
    df.dropna(inplace=True)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by date
    df.sort_values("Date", inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    save_path = os.path.join(PROCESSED_DIR, file)
    df.to_csv(save_path, index=False)

    print(f"Saved {save_path}")

print("\nPreprocessing completed!")