import pandas as pd
from pathlib import Path


# ============================================================
# BANK MARKETING PROJECT - DATA UNDERSTANDING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "bank_marketing_ingested.csv"


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
print("BANK MARKETING PROJECT - DATA UNDERSTANDING")
print("=" * 70)


# ============================================================
# 2. BUSINESS PROBLEM
# ============================================================

print("\n1. BUSINESS PROBLEM")
print(
    "Predict whether a bank customer will subscribe to a "
    "term deposit based on customer and campaign information."
)


# ============================================================
# 3. TARGET VARIABLE
# ============================================================

print("\n2. TARGET VARIABLE")

print("Target column: y")
print("Target meaning: Term deposit subscription")

print("\nTarget values:")
print(df["y"].value_counts())

print("\nTarget interpretation:")
print("yes = Customer subscribed to a term deposit")
print("no  = Customer did not subscribe to a term deposit")


# ============================================================
# 4. FEATURE GROUPS
# ============================================================

print("\n3. FEATURE GROUPS")


customer_features = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan"
]

campaign_features = [
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]

economic_features = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

call_feature = [
    "duration"
]


print("\nCustomer-related features:")
print(customer_features)

print("\nCampaign-related features:")
print(campaign_features)

print("\nEconomic/context features:")
print(economic_features)

print("\nCall-related feature:")
print(call_feature)


# ============================================================
# 5. CHECK FEATURE AVAILABILITY
# ============================================================

print("\n4. FEATURE AVAILABILITY CHECK")

all_expected_features = (
    customer_features
    + campaign_features
    + economic_features
    + call_feature
)

for feature in all_expected_features:
    if feature in df.columns:
        print(f"[OK] {feature}")
    else:
        print(f"[MISSING] {feature}")


# ============================================================
# 6. DATA TYPES
# ============================================================

print("\n5. DATA TYPE UNDERSTANDING")

numeric_features = df.select_dtypes(include="number").columns.tolist()
categorical_features = df.select_dtypes(include="object").columns.tolist()

print("\nNumerical columns:")
print(numeric_features)

print("\nCategorical columns:")
print(categorical_features)


# ============================================================
# 7. DURATION DATA LEAKAGE NOTE
# ============================================================

print("\n6. DURATION FEATURE - DATA LEAKAGE")

print(
    "duration represents the duration of the current marketing call."
)

print(
    "It is available only after the call has taken place."
)

print(
    "Therefore, duration will NOT be used as an input feature "
    "in the final predictive model."
)

print(
    "It may be examined during exploratory analysis, but it will "
    "be excluded before model training."
)


# ============================================================
# 8. CLASS IMBALANCE
# ============================================================

print("\n7. CLASS IMBALANCE")

target_distribution = (
    df["y"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(target_distribution)

print(
    "\nBecause the target classes are imbalanced, "
    "accuracy alone will not be sufficient for model evaluation."
)

print(
    "We will also evaluate precision, recall, F1-score and ROC-AUC."
)


# ============================================================
# 9. UNKNOWN VALUES
# ============================================================

print("\n8. UNKNOWN VALUES")

for column in df.select_dtypes(include="object").columns:

    unknown_count = (df[column] == "unknown").sum()

    if unknown_count > 0:

        percentage = (
            unknown_count / len(df) * 100
        )

        print(
            f"{column}: {unknown_count} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# 10. FINAL UNDERSTANDING
# ============================================================

print("\n9. FINAL DATA UNDERSTANDING")

print(
    f"Total observations: {df.shape[0]}"
)

print(
    f"Total columns: {df.shape[1]}"
)

print(
    f"Predictive input features before removing duration: "
    f"{len(all_expected_features)}"
)

print(
    "duration will be excluded from the final model."
)

print(
    "Target variable: y"
)

print(
    "Problem type: Supervised binary classification"
)


print("\n" + "=" * 70)
print("DATA UNDERSTANDING COMPLETE")
print("=" * 70)