import pandas as pd
from pathlib import Path


# ============================================================
# BANK MARKETING PROJECT
# FEATURE ENGINEERING PLAN
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "bank_marketing_ingested.csv"
OUTPUT_DIR = PROJECT_DIR / "eda_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE, sep=";")


print("=" * 75)
print("BANK MARKETING PROJECT - FEATURE ENGINEERING PLAN")
print("=" * 75)


# ============================================================
# 1. CURRENT FEATURES
# ============================================================

print("\n1. CURRENT FEATURE SET")
print("-" * 75)

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

print(f"Current model features: {len(model_features)}")

for feature in model_features:
    print(f"- {feature}")


# ============================================================
# 2. FEATURE ENGINEERING DECISIONS
# ============================================================

print("\n2. FEATURE ENGINEERING DECISIONS")
print("-" * 75)


feature_plan = {

    "age_group": {
        "source": "age",
        "action": "CREATE",
        "reason": (
            "Group customers into meaningful age ranges "
            "to capture possible nonlinear age effects."
        )
    },

    "campaign_intensity": {
        "source": "campaign",
        "action": "CREATE",
        "reason": (
            "Represent the number of contacts using meaningful "
            "contact-frequency groups."
        )
    },

    "previous_contact_history": {
        "source": "previous",
        "action": "CREATE",
        "reason": (
            "Distinguish customers with previous campaign "
            "experience from those without."
        )
    },

    "previous_contact_status": {
        "source": "pdays",
        "action": "CREATE",
        "reason": (
            "Convert pdays into a more interpretable indicator "
            "of whether the customer was previously contacted."
        )
    },

    "campaign_contact_ratio": {
        "source": "campaign + previous",
        "action": "CREATE",
        "reason": (
            "Capture relative campaign contact activity."
        )
    },

    "economic_environment": {
        "source": (
            "emp.var.rate + cons.price.idx + "
            "cons.conf.idx + euribor3m + nr.employed"
        ),
        "action": "KEEP",
        "reason": (
            "Economic context may contain useful predictive "
            "information and will be evaluated by the models."
        )
    },

    "duration": {
        "source": "duration",
        "action": "EXCLUDE",
        "reason": (
            "Duration is known only after the current call "
            "and therefore creates data leakage."
        )
    }
}


for name, details in feature_plan.items():

    print(f"\n{name}")
    print(f"  Source : {details['source']}")
    print(f"  Action : {details['action']}")
    print(f"  Reason : {details['reason']}")


# ============================================================
# 3. AGE GROUP
# ============================================================

print("\n3. AGE GROUP DISTRIBUTION")
print("-" * 75)

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 45, 55, 65, 100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65",
        "66+"
    ]
)

print(df["age_group"].value_counts().sort_index())


# ============================================================
# 4. CAMPAIGN INTENSITY
# ============================================================

print("\n4. CAMPAIGN INTENSITY")
print("-" * 75)

df["campaign_intensity"] = pd.cut(
    df["campaign"],
    bins=[0, 1, 2, 4, 999],
    labels=[
        "1 contact",
        "2 contacts",
        "3-4 contacts",
        "5+ contacts"
    ]
)

print(
    df["campaign_intensity"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 5. PREVIOUS CONTACT HISTORY
# ============================================================

print("\n5. PREVIOUS CONTACT HISTORY")
print("-" * 75)

df["previous_contact_history"] = pd.cut(
    df["previous"],
    bins=[-1, 0, 999],
    labels=[
        "No previous contact",
        "Previous contact"
    ]
)

print(
    df["previous_contact_history"]
    .value_counts()
)


# ============================================================
# 6. PREVIOUS CONTACT STATUS
# ============================================================

print("\n6. PREVIOUS CONTACT STATUS")
print("-" * 75)

df["previous_contact_status"] = (
    df["pdays"] != 999
).map({
    True: "Previously contacted",
    False: "Not previously contacted"
})

print(
    df["previous_contact_status"]
    .value_counts()
)


# ============================================================
# 7. CAMPAIGN CONTACT RATIO
# ============================================================

print("\n7. CAMPAIGN CONTACT RATIO")
print("-" * 75)

df["campaign_contact_ratio"] = (
    df["campaign"] /
    (df["previous"] + 1)
)

print(
    df["campaign_contact_ratio"]
    .describe()
    .round(2)
)


# ============================================================
# 8. TARGET RELATIONSHIP OF ENGINEERED FEATURES
# ============================================================

df["target"] = df["y"].map({
    "no": 0,
    "yes": 1
})


print("\n8. ENGINEERED FEATURE RELATIONSHIPS")
print("-" * 75)


age_group_rate = (
    df.groupby(
        "age_group",
        observed=True
    )["target"]
    .mean()
    .mul(100)
)

print("\nSubscription rate by age group:")
print(age_group_rate.round(2))


campaign_intensity_rate = (
    df.groupby(
        "campaign_intensity",
        observed=True
    )["target"]
    .mean()
    .mul(100)
)

print("\nSubscription rate by campaign intensity:")
print(campaign_intensity_rate.round(2))


previous_history_rate = (
    df.groupby(
        "previous_contact_history",
        observed=True
    )["target"]
    .mean()
    .mul(100)
)

print("\nSubscription rate by previous contact history:")
print(previous_history_rate.round(2))


previous_status_rate = (
    df.groupby(
        "previous_contact_status"
    )["target"]
    .mean()
    .mul(100)
)

print("\nSubscription rate by previous contact status:")
print(previous_status_rate.round(2))


# ============================================================
# 9. SAVE FEATURE ENGINEERING PLAN
# ============================================================

plan_rows = []

for feature_name, details in feature_plan.items():

    plan_rows.append({
        "feature": feature_name,
        "source": details["source"],
        "action": details["action"],
        "reason": details["reason"]
    })


plan_df = pd.DataFrame(plan_rows)

plan_file = (
    OUTPUT_DIR /
    "feature_engineering_plan.csv"
)

plan_df.to_csv(
    plan_file,
    index=False
)


# ============================================================
# 10. SAVE ENGINEERED FEATURE SUMMARY
# ============================================================

engineered_summary = pd.DataFrame({

    "feature": [
        "age_group",
        "campaign_intensity",
        "previous_contact_history",
        "previous_contact_status",
        "campaign_contact_ratio"
    ],

    "purpose": [
        "Capture age-group differences",
        "Capture campaign contact intensity",
        "Identify previous campaign experience",
        "Identify whether customer was previously contacted",
        "Capture relative contact activity"
    ]
})


engineered_summary.to_csv(
    OUTPUT_DIR /
    "engineered_feature_summary.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FEATURE ENGINEERING PLAN COMPLETE")
print("=" * 75)

print("\nPlanned engineered features:")
print("- age_group")
print("- campaign_intensity")
print("- previous_contact_history")
print("- previous_contact_status")
print("- campaign_contact_ratio")

print("\nExcluded:")
print("- duration")

print("\nSaved:")
print("- feature_engineering_plan.csv")
print("- engineered_feature_summary.csv")

print("\nNext stage:")
print("Build and compare multiple ML classification models.")