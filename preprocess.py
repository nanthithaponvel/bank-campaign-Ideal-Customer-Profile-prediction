import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from semantic_schema import (
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


# ============================================================
# BANK MARKETING PROJECT - PREPROCESSING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "bank_marketing_ingested.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}\n\n"
        "Run ingest.py first."
    )

df = pd.read_csv(DATA_FILE, sep=";")


print("=" * 70)
print("BANK MARKETING PROJECT - PREPROCESSING")
print("=" * 70)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df[MODEL_FEATURES].copy()
y = df["y"].map({
    "no": 0,
    "yes": 1
})


print("\n1. INPUT FEATURES")
print(f"Number of model features: {len(MODEL_FEATURES)}")

print("\nFeatures:")
for feature in MODEL_FEATURES:
    print(f"- {feature}")


# ============================================================
# 3. VERIFY DURATION IS NOT INCLUDED
# ============================================================

print("\n2. DATA LEAKAGE CHECK")

if "duration" in X.columns:
    raise ValueError(
        "ERROR: duration is present in model features. "
        "It must be excluded because of data leakage."
    )

print("[OK] duration is excluded from model features.")


# ============================================================
# 4. NUMERICAL PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 5. CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


# ============================================================
# 6. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# 7. VERIFY COLUMN GROUPS
# ============================================================

print("\n3. NUMERICAL FEATURES")
print(NUMERICAL_FEATURES)

print("\n4. CATEGORICAL FEATURES")
print(CATEGORICAL_FEATURES)

print(
    f"\nNumerical feature count: "
    f"{len(NUMERICAL_FEATURES)}"
)

print(
    f"Categorical feature count: "
    f"{len(CATEGORICAL_FEATURES)}"
)


# ============================================================
# 8. VERIFY ALL FEATURES ARE COVERED
# ============================================================

combined_features = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)

if set(combined_features) != set(MODEL_FEATURES):

    missing_from_preprocessing = [
        feature
        for feature in MODEL_FEATURES
        if feature not in combined_features
    ]

    extra_in_preprocessing = [
        feature
        for feature in combined_features
        if feature not in MODEL_FEATURES
    ]

    raise ValueError(
        "Preprocessing feature mismatch.\n"
        f"Missing: {missing_from_preprocessing}\n"
        f"Extra: {extra_in_preprocessing}"
    )

print(
    "\n[OK] All model features are covered by "
    "the preprocessing pipeline."
)


# ============================================================
# 9. FIT PREPROCESSOR FOR VERIFICATION
# ============================================================

print("\n5. FITTING PREPROCESSOR FOR VERIFICATION...")

X_transformed = preprocessor.fit_transform(X)

print("[OK] Preprocessor fitted successfully.")

print(
    f"Original feature count: {X.shape[1]}"
)

print(
    f"Transformed feature count: "
    f"{X_transformed.shape[1]}"
)


# ============================================================
# 10. TARGET VERIFICATION
# ============================================================

print("\n6. TARGET VERIFICATION")

print(
    f"Target values after encoding: "
    f"{sorted(y.dropna().unique())}"
)

if y.isnull().any():
    raise ValueError(
        "Target contains unexpected values."
    )

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Target must contain only 0 and 1 after encoding."
    )

print("[OK] Target encoded correctly: no = 0, yes = 1.")


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n7. PREPROCESSING SUMMARY")

print("Input:")
print("- Numerical features → median imputation + scaling")
print("- Categorical features → most-frequent imputation + one-hot encoding")
print("- Unknown categories during prediction → ignored safely")
print("- duration → excluded")

print("\nTarget:")
print("- no → 0")
print("- yes → 1")

print("\nIMPORTANT:")
print(
    "The fitted preprocessor will later be saved together "
    "with the trained model so prediction uses the exact "
    "same transformations."
)


print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)