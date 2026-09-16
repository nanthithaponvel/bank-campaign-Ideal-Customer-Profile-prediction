import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

from semantic_schema import MODEL_FEATURES, TARGET_COLUMN


# ============================================================
# BANK MARKETING PROJECT - TRAIN / TEST SPLIT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "bank_marketing_ingested.csv"

TRAIN_FILE = PROJECT_DIR / "data" / "train.csv"
TEST_FILE = PROJECT_DIR / "data" / "test.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATA_FILE}\n\n"
        "Run ingest.py first."
    )

df = pd.read_csv(DATA_FILE, sep=";")


print("=" * 70)
print("BANK MARKETING PROJECT - TRAIN / TEST SPLIT")
print("=" * 70)


# ============================================================
# 2. SELECT MODEL FEATURES AND TARGET
# ============================================================

X = df[MODEL_FEATURES].copy()

y = df[TARGET_COLUMN].map({
    "no": 0,
    "yes": 1
})


print("\n1. DATA PREPARED FOR SPLITTING")

print(f"Total observations: {len(df)}")
print(f"Model features: {len(MODEL_FEATURES)}")

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. DISPLAY SPLIT SIZES
# ============================================================

print("\n2. SPLIT RESULTS")

print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")

print(f"Training percentage: {len(X_train) / len(df) * 100:.2f}%")
print(f"Testing percentage : {len(X_test) / len(df) * 100:.2f}%")


# ============================================================
# 5. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n3. TARGET DISTRIBUTION CHECK")

train_distribution = (
    y_train.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

test_distribution = (
    y_test.value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\nTraining target distribution:")
print(train_distribution)

print("\nTesting target distribution:")
print(test_distribution)


# ============================================================
# 6. LEAKAGE CHECK
# ============================================================

print("\n4. DATA LEAKAGE CHECK")

if "duration" in X_train.columns:
    raise ValueError(
        "ERROR: duration is present in training features."
    )

if "duration" in X_test.columns:
    raise ValueError(
        "ERROR: duration is present in testing features."
    )

print("[OK] duration is excluded from training data.")
print("[OK] duration is excluded from testing data.")


# ============================================================
# 7. SAVE TRAINING DATA
# ============================================================

train_data = X_train.copy()
train_data[TARGET_COLUMN] = y_train

test_data = X_test.copy()
test_data[TARGET_COLUMN] = y_test


train_data.to_csv(TRAIN_FILE, sep=";", index=False)
test_data.to_csv(TEST_FILE, sep=";", index=False)


# ============================================================
# 8. VERIFY SAVED FILES
# ============================================================

print("\n5. SAVED DATA")

print(f"Training data saved to:")
print(TRAIN_FILE)

print(f"\nTesting data saved to:")
print(TEST_FILE)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n6. FINAL SUMMARY")

print("Train/test split: 80/20")
print("Split method: Stratified")
print("Random state: 42")
print("Target encoding: no = 0, yes = 1")

print(
    "\nThe training and testing datasets are ready "
    "for the next stages of the project."
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT COMPLETE")
print("=" * 70)