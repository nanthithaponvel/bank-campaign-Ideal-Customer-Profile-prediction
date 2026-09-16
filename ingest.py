import pandas as pd
from pathlib import Path


# ============================================================
# BANK MARKETING PROJECT - DATA INGESTION
# ============================================================

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# Input dataset
INPUT_FILE = DATA_DIR / "bank-additional-full.csv"

# Output dataset
OUTPUT_FILE = DATA_DIR / "bank_marketing_ingested.csv"


# ============================================================
# 1. CHECK THAT THE DATASET EXISTS
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{INPUT_FILE}\n\n"
        "Make sure bank-additional-full.csv is inside the data folder."
    )


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("BANK MARKETING PROJECT - DATA INGESTION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE, sep=";")


# ============================================================
# 3. BASIC VERIFICATION
# ============================================================

print("\nDataset loaded successfully!")

print("\nDataset shape:")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumn names:")
print(list(df.columns))


# ============================================================
# 4. TARGET CHECK
# ============================================================

if "y" not in df.columns:
    raise ValueError("Target column 'y' was not found in the dataset.")

print("\nTarget column:")
print("y")

print("\nTarget values:")
print(df["y"].value_counts())


# ============================================================
# 5. SAVE INGESTED DATASET
# ============================================================

df.to_csv(OUTPUT_FILE, sep=";", index=False)

print("\nIngested dataset saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("DATA INGESTION COMPLETE")
print("=" * 70)