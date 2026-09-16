import os
import pandas as pd
import numpy as np
import joblib

# ================================================================
# BANK MARKETING PROJECT
# FULL DATASET CUSTOMER SCORING
# ================================================================

print("=" * 75)
print("BANK MARKETING PROJECT - FULL DATASET CUSTOMER SCORING")
print("=" * 75)


# ================================================================
# 1. PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bank-additional-full.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.joblib"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_features.txt"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 2. LOAD MODEL
# ================================================================

print("\n1. MODEL LOADING")
print("-" * 75)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("[OK] best_model.joblib loaded.")
print("Model: Random Forest")


# ================================================================
# 3. LOAD FULL DATASET
# ================================================================

print("\n2. FULL DATASET LOADING")
print("-" * 75)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH, sep=";")

print(f"Total observations: {len(df):,}")
print(f"Total columns: {len(df.columns)}")


# ================================================================
# 4. LOAD MODEL FEATURES
# ================================================================

print("\n3. MODEL FEATURES")
print("-" * 75)

if os.path.exists(FEATURES_PATH):

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        model_features = [
            line.strip()
            for line in f
            if line.strip()
        ]

else:

    # Fallback to the known 19 features
    model_features = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed"
    ]

print(f"Number of model features: {len(model_features)}")

for feature in model_features:
    print(f"- {feature}")


# ================================================================
# 5. DATA VALIDATION
# ================================================================

print("\n4. DATA VALIDATION")
print("-" * 75)

missing_features = [
    feature
    for feature in model_features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "The following model features are missing from the dataset:\n"
        + "\n".join(missing_features)
    )

print("[OK] All 19 model features are available.")


# ================================================================
# 6. DURATION LEAKAGE CHECK
# ================================================================

print("\n5. DATA LEAKAGE CHECK")
print("-" * 75)

if "duration" in model_features:
    raise ValueError(
        "ERROR: duration is present in model features."
    )

print("[OK] duration is excluded from prediction.")
print("Reason: duration is only known after the marketing call.")


# ================================================================
# 7. PREPARE INPUT DATA
# ================================================================

print("\n6. PREDICTION DATA PREPARATION")
print("-" * 75)

X_full = df[model_features].copy()

print(f"Prediction input shape: {X_full.shape}")

if len(X_full.columns) != 19:
    raise ValueError(
        f"Expected 19 features, found {len(X_full.columns)}."
    )

print("[OK] Prediction data prepared.")


# ================================================================
# 8. GENERATE PROBABILITIES
# ================================================================

print("\n7. GENERATING SUBSCRIPTION PROBABILITIES")
print("-" * 75)

if not hasattr(model, "predict_proba"):
    raise ValueError(
        "The selected model does not support probability prediction."
    )

probabilities = model.predict_proba(X_full)[:, 1]

print("[OK] Subscription probabilities generated.")


# ================================================================
# 9. GENERATE PREDICTIONS
# ================================================================

print("\n8. GENERATING CUSTOMER PREDICTIONS")
print("-" * 75)

predictions = (probabilities >= 0.50).astype(int)

print("[OK] Customer predictions generated.")


# ================================================================
# 10. CREATE RESULT DATAFRAME
# ================================================================

results = df.copy()

results["subscription_probability"] = probabilities

results["predicted_subscription"] = np.where(
    predictions == 1,
    "yes",
    "no"
)


# ================================================================
# 11. CAMPAIGN PRIORITY
# ================================================================
#
# High   >= 70%
# Medium 40% - 69.99%
# Low    < 40%
#
# These are business prioritization thresholds,
# not model probability thresholds.
# ================================================================

def assign_priority(probability):

    if probability >= 0.70:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    else:
        return "Low"


results["campaign_priority"] = (
    results["subscription_probability"]
    .apply(assign_priority)
)


# ================================================================
# 12. RANK CUSTOMERS
# ================================================================

results = results.sort_values(
    by="subscription_probability",
    ascending=False
).reset_index(drop=True)

results["customer_rank"] = (
    results.index + 1
)


# ================================================================
# 13. ROUND PROBABILITY
# ================================================================

results["subscription_probability"] = (
    results["subscription_probability"]
    .round(6)
)


# ================================================================
# 14. REORDER IMPORTANT COLUMNS
# ================================================================

priority_columns = [
    "customer_rank",
    "subscription_probability",
    "predicted_subscription",
    "campaign_priority"
]

other_columns = [
    column
    for column in results.columns
    if column not in priority_columns
]

results = results[
    priority_columns + other_columns
]


# ================================================================
# 15. SAVE COMPLETE SCORING FILE
# ================================================================

full_output_path = os.path.join(
    OUTPUT_DIR,
    "full_customer_predictions.csv"
)

results.to_csv(
    full_output_path,
    index=False
)

print("\n9. FULL CUSTOMER SCORING SAVED")
print("-" * 75)

print(f"File: {full_output_path}")


# ================================================================
# 16. HIGH PRIORITY CUSTOMERS
# ================================================================

high_priority = results[
    results["campaign_priority"] == "High"
].copy()

high_priority_path = os.path.join(
    OUTPUT_DIR,
    "full_high_priority_customers.csv"
)

high_priority.to_csv(
    high_priority_path,
    index=False
)


# ================================================================
# 17. CAMPAIGN TARGET LIST
# ================================================================

campaign_target_list = results[
    results["campaign_priority"].isin(
        ["High", "Medium"]
    )
].copy()

campaign_target_path = os.path.join(
    OUTPUT_DIR,
    "full_campaign_target_list.csv"
)

campaign_target_list.to_csv(
    campaign_target_path,
    index=False
)


# ================================================================
# 18. SUMMARY
# ================================================================

print("\n10. FULL DATASET PREDICTION SUMMARY")
print("-" * 75)

total_customers = len(results)

predicted_subscribers = (
    results["predicted_subscription"]
    .eq("yes")
    .sum()
)

predicted_non_subscribers = (
    results["predicted_subscription"]
    .eq("no")
    .sum()
)

average_probability = (
    results["subscription_probability"]
    .mean()
)

high_count = (
    results["campaign_priority"]
    .eq("High")
    .sum()
)

medium_count = (
    results["campaign_priority"]
    .eq("Medium")
    .sum()
)

low_count = (
    results["campaign_priority"]
    .eq("Low")
    .sum()
)


print(f"Total customers scored       : {total_customers:,}")
print(
    f"Predicted subscribers        : {predicted_subscribers:,}"
)
print(
    f"Predicted non-subscribers    : {predicted_non_subscribers:,}"
)
print(
    f"Average probability          : {average_probability:.2%}"
)

print("\nCAMPAIGN PRIORITY")
print("-" * 75)

print(
    f"High priority   : {high_count:,} "
    f"({high_count / total_customers:.2%})"
)

print(
    f"Medium priority : {medium_count:,} "
    f"({medium_count / total_customers:.2%})"
)

print(
    f"Low priority    : {low_count:,} "
    f"({low_count / total_customers:.2%})"
)


# ================================================================
# 19. TOP 20 CUSTOMERS
# ================================================================

print("\n11. TOP 20 CUSTOMERS")
print("-" * 75)

top20 = results[
    [
        "customer_rank",
        "subscription_probability",
        "predicted_subscription",
        "campaign_priority"
    ]
].head(20)

print(
    top20.to_string(index=False)
)


# ================================================================
# 20. TOP 10% ANALYSIS
# ================================================================

top_10_count = max(
    1,
    int(np.ceil(total_customers * 0.10))
)

top_10 = results.head(top_10_count)

print("\n12. TOP 10% CUSTOMER ANALYSIS")
print("-" * 75)

print(f"Top 10% customers: {len(top_10):,}")
print(
    f"Average probability: "
    f"{top_10['subscription_probability'].mean():.2%}"
)

print(
    f"Predicted subscribers: "
    f"{(top_10['predicted_subscription'] == 'yes').sum():,}"
)


# ================================================================
# 21. SAVE TOP 10%
# ================================================================

top_10_path = os.path.join(
    OUTPUT_DIR,
    "full_top_10_percent_customers.csv"
)

top_10.to_csv(
    top_10_path,
    index=False
)


# ================================================================
# 22. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 75)
print("FULL DATASET CUSTOMER SCORING COMPLETE")
print("=" * 75)

print("\nGenerated files:")

print("- models/full_customer_predictions.csv")
print("- models/full_high_priority_customers.csv")
print("- models/full_campaign_target_list.csv")
print("- models/full_top_10_percent_customers.csv")

print("\nIMPORTANT:")
print(
    "These are propensity scores generated for the available "
    "historical dataset."
)

print(
    "They should be presented as customer prioritization scores, "
    "not guaranteed future outcomes."
)

print("=" * 75)
