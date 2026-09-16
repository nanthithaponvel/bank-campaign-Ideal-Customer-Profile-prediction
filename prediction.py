import pandas as pd
import numpy as np
from pathlib import Path

import joblib


# ============================================================
# BANK MARKETING PROJECT
# CUSTOMER-LEVEL PREDICTION SYSTEM
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

TEST_FILE = PROJECT_DIR / "data" / "test.csv"
MODEL_FILE = PROJECT_DIR / "models" / "best_model.joblib"
OUTPUT_DIR = PROJECT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)


print("=" * 75)
print("BANK MARKETING PROJECT - CUSTOMER-LEVEL PREDICTION")
print("=" * 75)


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found:\n{TEST_FILE}\n\n"
        "Run split.py first."
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_FILE}\n\n"
        "Run train_models.py first."
    )


# ============================================================
# 2. LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)


print("\n1. MODEL LOADED")
print("-" * 75)

print(
    "Model: Random Forest"
)

print(
    "[OK] best_model.joblib loaded."
)


# ============================================================
# 3. LOAD CUSTOMER DATA
# ============================================================

df = pd.read_csv(
    TEST_FILE,
    sep=";"
)


print("\n2. CUSTOMER DATA LOADED")
print("-" * 75)

print(
    f"Customers available: {len(df)}"
)


# ============================================================
# 4. LOAD MODEL FEATURES
# ============================================================

features_file = (
    OUTPUT_DIR /
    "model_features.txt"
)


if not features_file.exists():
    raise FileNotFoundError(
        "model_features.txt not found."
    )


with open(
    features_file,
    "r",
    encoding="utf-8"
) as file:

    lines = file.readlines()


MODEL_FEATURES = []

for line in lines:

    line = line.strip()

    if (
        line
        and line != "MODEL FEATURES"
        and line != "==============="
    ):
        MODEL_FEATURES.append(line)


print("\n3. MODEL FEATURES LOADED")
print("-" * 75)

print(
    f"Number of model features: "
    f"{len(MODEL_FEATURES)}"
)

for feature in MODEL_FEATURES:
    print(f"- {feature}")


# ============================================================
# 5. CHECK DURATION IS NOT USED
# ============================================================

if "duration" in MODEL_FEATURES:

    raise ValueError(
        "ERROR: duration must not be used "
        "for future campaign prediction."
    )


print(
    "[OK] duration is excluded."
)


# ============================================================
# 6. PREPARE CUSTOMER DATA
# ============================================================

X = df[
    MODEL_FEATURES
].copy()


print("\n4. CUSTOMER DATA PREPARED")
print("-" * 75)

print(
    f"Prediction input shape: {X.shape}"
)


# ============================================================
# 7. GENERATE SUBSCRIPTION PROBABILITIES
# ============================================================

print("\n5. GENERATING CUSTOMER PREDICTIONS")
print("-" * 75)

probabilities = (
    model
    .predict_proba(X)[:, 1]
)


predictions = (
    model
    .predict(X)
)


print(
    "[OK] Subscription probabilities generated."
)

print(
    "[OK] Customer predictions generated."
)


# ============================================================
# 8. CREATE CUSTOMER PREDICTION DATAFRAME
# ============================================================

prediction_df = df.copy()


prediction_df[
    "subscription_probability"
] = probabilities


# FIX:
# Convert NumPy prediction array into a Pandas Series
# before using .map()

prediction_df[
    "predicted_subscription"
] = pd.Series(
    predictions,
    index=prediction_df.index
).astype(int).map({
    0: "no",
    1: "yes"
})


# ============================================================
# 9. CREATE PRIORITY LEVEL
# ============================================================

def assign_priority(probability):

    if probability >= 0.70:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    else:
        return "Low"


prediction_df[
    "campaign_priority"
] = (
    prediction_df[
        "subscription_probability"
    ]
    .apply(assign_priority)
)


# ============================================================
# 10. CREATE CUSTOMER RANK
# ============================================================

prediction_df[
    "customer_rank"
] = (
    prediction_df[
        "subscription_probability"
    ]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)


# ============================================================
# 11. SORT CUSTOMERS BY PROBABILITY
# ============================================================

prediction_df = (
    prediction_df
    .sort_values(
        by="subscription_probability",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 12. DISPLAY TOP 20 CUSTOMERS
# ============================================================

print("\n" + "=" * 75)

print(
    "TOP 20 CUSTOMERS TO PRIORITIZE"
)

print("=" * 75)


display_columns = [
    "customer_rank",
    "subscription_probability",
    "predicted_subscription",
    "campaign_priority"
]


print(
    prediction_df[
        display_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 13. PRIORITY DISTRIBUTION
# ============================================================

priority_counts = (
    prediction_df[
        "campaign_priority"
    ]
    .value_counts()
    .reindex(
        [
            "High",
            "Medium",
            "Low"
        ],
        fill_value=0
    )
)


print("\n" + "=" * 75)

print(
    "CAMPAIGN PRIORITY DISTRIBUTION"
)

print("=" * 75)


for priority, count in priority_counts.items():

    percentage = (
        count /
        len(prediction_df)
        * 100
    )

    print(
        f"{priority:<10}: "
        f"{count:>5} customers "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 14. PREDICTED SUBSCRIPTION DISTRIBUTION
# ============================================================

predicted_counts = (
    prediction_df[
        "predicted_subscription"
    ]
    .value_counts()
)


print("\n" + "=" * 75)

print(
    "PREDICTED SUBSCRIPTION DISTRIBUTION"
)

print("=" * 75)


print(
    predicted_counts
)


# ============================================================
# 15. PROBABILITY SUMMARY
# ============================================================

print("\n" + "=" * 75)

print(
    "SUBSCRIPTION PROBABILITY SUMMARY"
)

print("=" * 75)


print(
    f"Average probability: "
    f"{prediction_df['subscription_probability'].mean():.4f}"
)

print(
    f"Minimum probability: "
    f"{prediction_df['subscription_probability'].min():.4f}"
)

print(
    f"Maximum probability: "
    f"{prediction_df['subscription_probability'].max():.4f}"
)


# ============================================================
# 16. SAVE ALL CUSTOMER PREDICTIONS
# ============================================================

all_predictions_file = (
    OUTPUT_DIR /
    "customer_predictions.csv"
)


prediction_df.to_csv(
    all_predictions_file,
    index=False
)


# ============================================================
# 17. SAVE HIGH PRIORITY CUSTOMERS
# ============================================================

high_priority_df = (
    prediction_df[
        prediction_df[
            "campaign_priority"
        ] == "High"
    ]
    .copy()
)


high_priority_file = (
    OUTPUT_DIR /
    "high_priority_customers.csv"
)


high_priority_df.to_csv(
    high_priority_file,
    index=False
)


# ============================================================
# 18. CREATE CAMPAIGN TARGET LIST
# ============================================================

campaign_columns = [
    "customer_rank",
    "age",
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "contact",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "subscription_probability",
    "predicted_subscription",
    "campaign_priority"
]


campaign_columns = [
    column
    for column in campaign_columns
    if column in prediction_df.columns
]


campaign_list = (
    prediction_df[
        campaign_columns
    ]
    .copy()
)


campaign_file = (
    OUTPUT_DIR /
    "campaign_target_list.csv"
)


campaign_list.to_csv(
    campaign_file,
    index=False
)


# ============================================================
# 19. DISPLAY HIGH PRIORITY SUMMARY
# ============================================================

print("\n" + "=" * 75)

print(
    "HIGH PRIORITY CUSTOMER SUMMARY"
)

print("=" * 75)


if len(high_priority_df) > 0:

    print(
        f"High priority customers: "
        f"{len(high_priority_df)}"
    )

    print(
        f"Highest probability: "
        f"{high_priority_df['subscription_probability'].max():.4f}"
    )

    print(
        f"Average probability among high priority: "
        f"{high_priority_df['subscription_probability'].mean():.4f}"
    )

else:

    print(
        "No high-priority customers found."
    )


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)

print(
    "CUSTOMER-LEVEL PREDICTION COMPLETE"
)

print("=" * 75)


print(
    f"\nTotal customers scored: "
    f"{len(prediction_df)}"
)


print(
    f"High priority customers: "
    f"{priority_counts['High']}"
)


print(
    f"Medium priority customers: "
    f"{priority_counts['Medium']}"
)


print(
    f"Low priority customers: "
    f"{priority_counts['Low']}"
)


print("\nGenerated files:")

print(
    "- models/customer_predictions.csv"
)

print(
    "- models/high_priority_customers.csv"
)

print(
    "- models/campaign_target_list.csv"
)


print("\nBusiness purpose:")

print(
    "Customers are ranked according to their "
    "predicted probability of subscribing to "
    "a term deposit."
)


print(
    "\nThe bank can use the ranking to prioritize "
    "customers for future marketing campaigns."
)


print(
    "\n" + "=" * 75
)