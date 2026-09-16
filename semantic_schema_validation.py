import pandas as pd
from pathlib import Path

from semantic_schema import (
    TARGET_COLUMN,
    EXCLUDED_FEATURES,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXPECTED_COLUMNS,
)


# ============================================================
# BANK MARKETING PROJECT - SEMANTIC SCHEMA VALIDATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "bank_marketing_ingested.csv"


print("=" * 70)
print("SEMANTIC SCHEMA VALIDATION")
print("=" * 70)


# ============================================================
# 1. CHECK DATASET
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}\n\n"
        "Run ingest.py first."
    )

df = pd.read_csv(DATA_FILE, sep=";")


# ============================================================
# 2. ACTUAL COLUMNS
# ============================================================

actual_columns = df.columns.tolist()


print("\n1. ACTUAL DATASET COLUMNS")
print(actual_columns)


# ============================================================
# 3. EXPECTED COLUMNS
# ============================================================

print("\n2. EXPECTED DATASET COLUMNS")
print(EXPECTED_COLUMNS)


# ============================================================
# 4. CHECK FOR MISSING COLUMNS
# ============================================================

missing_columns = [
    column
    for column in EXPECTED_COLUMNS
    if column not in actual_columns
]


print("\n3. MISSING EXPECTED COLUMNS")

if missing_columns:
    for column in missing_columns:
        print(f"[MISSING] {column}")
else:
    print("[OK] No expected columns are missing.")


# ============================================================
# 5. CHECK FOR UNEXPECTED COLUMNS
# ============================================================

unexpected_columns = [
    column
    for column in actual_columns
    if column not in EXPECTED_COLUMNS
]


print("\n4. UNEXPECTED COLUMNS")

if unexpected_columns:
    for column in unexpected_columns:
        print(f"[UNEXPECTED] {column}")
else:
    print("[OK] No unexpected columns found.")


# ============================================================
# 6. TARGET VALIDATION
# ============================================================

print("\n5. TARGET VALIDATION")

if TARGET_COLUMN in df.columns:
    print(f"[OK] Target column '{TARGET_COLUMN}' exists.")
else:
    print(f"[ERROR] Target column '{TARGET_COLUMN}' is missing.")


# ============================================================
# 7. EXCLUDED FEATURE VALIDATION
# ============================================================

print("\n6. EXCLUDED FEATURE VALIDATION")

for feature in EXCLUDED_FEATURES:

    if feature in df.columns:
        print(
            f"[OK] '{feature}' exists in the dataset "
            "but is excluded from model features."
        )
    else:
        print(f"[ERROR] Excluded feature '{feature}' is missing.")


# ============================================================
# 8. MODEL FEATURE VALIDATION
# ============================================================

print("\n7. MODEL FEATURE VALIDATION")

missing_model_features = [
    feature
    for feature in MODEL_FEATURES
    if feature not in df.columns
]

if missing_model_features:
    for feature in missing_model_features:
        print(f"[MISSING] {feature}")
else:
    print(
        f"[OK] All {len(MODEL_FEATURES)} model features "
        "are present."
    )


# ============================================================
# 9. NUMERICAL FEATURE VALIDATION
# ============================================================

print("\n8. NUMERICAL FEATURE VALIDATION")

actual_numeric = set(
    df[MODEL_FEATURES]
    .select_dtypes(include="number")
    .columns
)

expected_numeric = set(NUMERICAL_FEATURES)

if actual_numeric == expected_numeric:
    print("[OK] Numerical feature schema matches.")
else:
    print("Expected numerical features:")
    print(sorted(expected_numeric))

    print("\nActual numerical features:")
    print(sorted(actual_numeric))

    print(
        "\n[WARNING] Numerical feature schema does not match."
    )


# ============================================================
# 10. CATEGORICAL FEATURE VALIDATION
# ============================================================

print("\n9. CATEGORICAL FEATURE VALIDATION")

actual_categorical = set(
    df[MODEL_FEATURES]
    .select_dtypes(include="object")
    .columns
)

expected_categorical = set(CATEGORICAL_FEATURES)

if actual_categorical == expected_categorical:
    print("[OK] Categorical feature schema matches.")
else:
    print("Expected categorical features:")
    print(sorted(expected_categorical))

    print("\nActual categorical features:")
    print(sorted(actual_categorical))

    print(
        "\n[WARNING] Categorical feature schema does not match."
    )


# ============================================================
# 11. FEATURE COUNT
# ============================================================

print("\n10. FEATURE COUNT")

print(f"Model features: {len(MODEL_FEATURES)}")
print(f"Numerical features: {len(NUMERICAL_FEATURES)}")
print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
print(f"Excluded features: {len(EXCLUDED_FEATURES)}")


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

print("\n11. FINAL VALIDATION")

validation_passed = True

if missing_columns:
    validation_passed = False

if unexpected_columns:
    validation_passed = False

if TARGET_COLUMN not in df.columns:
    validation_passed = False

if any(feature not in df.columns for feature in EXCLUDED_FEATURES):
    validation_passed = False

if missing_model_features:
    validation_passed = False


if validation_passed:
    print("\n[PASS] Semantic schema validation successful.")
    print("The dataset matches the defined project schema.")
else:
    print("\n[FAIL] Semantic schema validation failed.")
    print("Review the issues listed above.")


print("\n" + "=" * 70)
print("SEMANTIC SCHEMA VALIDATION COMPLETE")
print("=" * 70)