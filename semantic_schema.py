# ============================================================
# BANK MARKETING PROJECT - SEMANTIC SCHEMA
# ============================================================

"""
This file is the single source of truth for the dataset schema.

IMPORTANT:
- 'y' is the target variable.
- 'duration' is intentionally excluded from model features
  because it causes data leakage.
- The same feature lists will be reused during preprocessing,
  training, prediction, and Streamlit.
"""


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "y"


# ============================================================
# FEATURES EXCLUDED FROM THE FINAL MODEL
# ============================================================

EXCLUDED_FEATURES = [
    "duration"
]


# ============================================================
# CUSTOMER FEATURES
# ============================================================

CUSTOMER_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan"
]


# ============================================================
# CAMPAIGN FEATURES
# ============================================================

CAMPAIGN_FEATURES = [
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]


# ============================================================
# ECONOMIC FEATURES
# ============================================================

ECONOMIC_FEATURES = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]


# ============================================================
# FINAL MODEL FEATURES
# ============================================================

MODEL_FEATURES = (
    CUSTOMER_FEATURES
    + CAMPAIGN_FEATURES
    + ECONOMIC_FEATURES
)


# ============================================================
# NUMERICAL FEATURES
# ============================================================

NUMERICAL_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome"
]


# ============================================================
# TARGET VALUES
# ============================================================

TARGET_MAPPING = {
    "no": 0,
    "yes": 1
}


# ============================================================
# EXPECTED DATASET COLUMNS
# ============================================================

EXPECTED_COLUMNS = MODEL_FEATURES + EXCLUDED_FEATURES + [TARGET_COLUMN]


# ============================================================
# DISPLAY INFORMATION
# ============================================================

FEATURE_DESCRIPTIONS = {
    "age": "Age of the customer",
    "job": "Type of job",
    "marital": "Marital status",
    "education": "Education level",
    "default": "Credit in default",
    "housing": "Housing loan",
    "loan": "Personal loan",
    "contact": "Contact communication type",
    "month": "Month of last contact",
    "day_of_week": "Day of week of last contact",
    "campaign": "Number of contacts during this campaign",
    "pdays": "Days since customer was last contacted",
    "previous": "Number of contacts before this campaign",
    "poutcome": "Outcome of previous marketing campaign",
    "emp.var.rate": "Employment variation rate",
    "cons.price.idx": "Consumer price index",
    "cons.conf.idx": "Consumer confidence index",
    "euribor3m": "Euribor 3 month rate",
    "nr.employed": "Number of employees"
}


# ============================================================
# BASIC SCHEMA INFORMATION
# ============================================================

SCHEMA_VERSION = "1.0"

print("=" * 70)
print("BANK MARKETING SEMANTIC SCHEMA")
print("=" * 70)

print(f"\nTarget column: {TARGET_COLUMN}")

print(f"\nTotal model features: {len(MODEL_FEATURES)}")

print("\nModel features:")
for feature in MODEL_FEATURES:
    print(f"- {feature}")

print("\nExcluded features:")
for feature in EXCLUDED_FEATURES:
    print(f"- {feature}")

print("\nNumerical features:")
print(NUMERICAL_FEATURES)

print("\nCategorical features:")
print(CATEGORICAL_FEATURES)

print("\nTarget mapping:")
print(TARGET_MAPPING)

print(f"\nSchema version: {SCHEMA_VERSION}")

print("\n" + "=" * 70)
print("SEMANTIC SCHEMA DEFINED")
print("=" * 70)
