import pandas as pd
from pathlib import Path


# ============================================================
# BANK MARKETING PROJECT - DATA PROFILING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "bank_marketing_ingested.csv"


# ============================================================
# 1. CHECK FILE
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATA_FILE}\n\n"
        "Run ingest.py first."
    )


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_FILE, sep=";")


print("=" * 70)
print("BANK MARKETING PROJECT - DATA PROFILE")
print("=" * 70)


# ============================================================
# 3. DATASET SHAPE
# ============================================================

print("\n1. DATASET SHAPE")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 4. COLUMN INFORMATION
# ============================================================

print("\n2. COLUMN INFORMATION")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

print(column_info.to_string(index=False))


# ============================================================
# 5. DUPLICATE ROWS
# ============================================================

print("\n3. DUPLICATE ROWS")
print(f"Duplicate rows: {df.duplicated().sum()}")


# ============================================================
# 6. NUMERICAL FEATURES
# ============================================================

print("\n4. NUMERICAL FEATURES")

numeric_columns = df.select_dtypes(include="number").columns.tolist()

print(numeric_columns)

if numeric_columns:
    print("\nNumerical summary:")
    print(df[numeric_columns].describe().round(2).to_string())


# ============================================================
# 7. CATEGORICAL FEATURES
# ============================================================

print("\n5. CATEGORICAL FEATURES")

categorical_columns = df.select_dtypes(include="object").columns.tolist()

print(categorical_columns)


# ============================================================
# 8. CATEGORICAL VALUE COUNTS
# ============================================================

print("\n6. CATEGORICAL VALUE COUNTS")

for column in categorical_columns:
    print("\n" + "-" * 60)
    print(f"{column}")
    print("-" * 60)
    print(df[column].value_counts(dropna=False).to_string())


# ============================================================
# 9. TARGET PROFILE
# ============================================================

print("\n7. TARGET VARIABLE PROFILE")

if "y" in df.columns:
    print("Target column: y")

    print("\nTarget counts:")
    print(df["y"].value_counts(dropna=False).to_string())

    print("\nTarget percentages:")
    print(
        (df["y"].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
        .to_string()
    )
else:
    print("WARNING: Target column 'y' was not found.")


# ============================================================
# 10. UNKNOWN VALUES
# ============================================================

print("\n8. 'UNKNOWN' VALUES")

unknown_summary = {}

for column in categorical_columns:
    count = (df[column] == "unknown").sum()

    if count > 0:
        unknown_summary[column] = count

if unknown_summary:
    for column, count in unknown_summary.items():
        print(f"{column}: {count}")
else:
    print("No 'unknown' values found.")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DATA PROFILING COMPLETE")
print("=" * 70)
