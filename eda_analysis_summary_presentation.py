import pandas as pd
from pathlib import Path


# ============================================================
# BANK MARKETING PROJECT
# EDA ANALYSIS, SUMMARY & PRESENTATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "bank_marketing_ingested.csv"
OUTPUT_DIR = PROJECT_DIR / "eda_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE, sep=";")


# Create binary target
df["target"] = df["y"].map({
    "no": 0,
    "yes": 1
})


print("=" * 75)
print("BANK MARKETING PROJECT")
print("EDA ANALYSIS, SUMMARY & PRESENTATION")
print("=" * 75)


# ============================================================
# 2. OVERALL BUSINESS RESULT
# ============================================================

overall_rate = df["target"].mean() * 100

print("\n1. OVERALL BUSINESS RESULT")
print("-" * 75)

print(
    f"Overall term deposit subscription rate: "
    f"{overall_rate:.2f}%"
)

print(
    f"Customers who subscribed: "
    f"{(df['target'] == 1).sum():,}"
)

print(
    f"Customers who did not subscribe: "
    f"{(df['target'] == 0).sum():,}"
)


# ============================================================
# 3. JOB ANALYSIS
# ============================================================

job_rate = (
    df.groupby("job")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n2. JOB ANALYSIS")
print("-" * 75)

print(job_rate.round(2))

print(
    f"\nHighest subscription-rate job category: "
    f"{job_rate.index[0]} ({job_rate.iloc[0]:.2f}%)"
)

print(
    f"Lowest subscription-rate job category: "
    f"{job_rate.index[-1]} ({job_rate.iloc[-1]:.2f}%)"
)


# ============================================================
# 4. EDUCATION ANALYSIS
# ============================================================

education_rate = (
    df.groupby("education")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n3. EDUCATION ANALYSIS")
print("-" * 75)

print(education_rate.round(2))

print(
    f"\nHighest subscription-rate education group: "
    f"{education_rate.index[0]} ({education_rate.iloc[0]:.2f}%)"
)


# ============================================================
# 5. CONTACT METHOD ANALYSIS
# ============================================================

contact_rate = (
    df.groupby("contact")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n4. CONTACT METHOD ANALYSIS")
print("-" * 75)

print(contact_rate.round(2))

print(
    f"\nBest-performing contact method: "
    f"{contact_rate.index[0]} ({contact_rate.iloc[0]:.2f}%)"
)


# ============================================================
# 6. MONTH ANALYSIS
# ============================================================

month_rate = (
    df.groupby("month")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n5. MONTH ANALYSIS")
print("-" * 75)

print(month_rate.round(2))

print(
    f"\nHighest subscription-rate month: "
    f"{month_rate.index[0]} ({month_rate.iloc[0]:.2f}%)"
)


# ============================================================
# 7. PREVIOUS CAMPAIGN OUTCOME
# ============================================================

previous_rate = (
    df.groupby("poutcome")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n6. PREVIOUS CAMPAIGN OUTCOME")
print("-" * 75)

print(previous_rate.round(2))

print(
    f"\nBest previous campaign outcome group: "
    f"{previous_rate.index[0]} ({previous_rate.iloc[0]:.2f}%)"
)


# ============================================================
# 8. CAMPAIGN CONTACT ANALYSIS
# ============================================================

campaign_rate = (
    df.groupby("campaign")["target"]
    .mean()
    .mul(100)
)

print("\n7. CAMPAIGN CONTACT ANALYSIS")
print("-" * 75)

print(campaign_rate.head(15).round(2))


# ============================================================
# 9. HOUSING LOAN
# ============================================================

housing_rate = (
    df.groupby("housing")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n8. HOUSING LOAN ANALYSIS")
print("-" * 75)

print(housing_rate.round(2))


# ============================================================
# 10. PERSONAL LOAN
# ============================================================

loan_rate = (
    df.groupby("loan")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n9. PERSONAL LOAN ANALYSIS")
print("-" * 75)

print(loan_rate.round(2))


# ============================================================
# 11. ECONOMIC FACTOR ANALYSIS
# ============================================================

economic_features = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

economic_corr = (
    df[economic_features + ["target"]]
    .corr()["target"]
    .drop("target")
    .sort_values(
        key=lambda x: abs(x),
        ascending=False
    )
)

print("\n10. ECONOMIC FACTOR ANALYSIS")
print("-" * 75)

print(economic_corr.round(3))

print(
    f"\nStrongest absolute correlation with subscription: "
    f"{economic_corr.index[0]} "
    f"({economic_corr.iloc[0]:.3f})"
)


# ============================================================
# 12. AGE ANALYSIS
# ============================================================

print("\n11. AGE ANALYSIS")
print("-" * 75)

print(
    df.groupby("y")["age"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)


# ============================================================
# 13. UNKNOWN VALUE ANALYSIS
# ============================================================

print("\n12. UNKNOWN VALUE ANALYSIS")
print("-" * 75)

unknown_results = []

for column in df.columns:

    if df[column].dtype == "object":

        count = (df[column] == "unknown").sum()

        if count > 0:

            percentage = count / len(df) * 100

            unknown_results.append({
                "feature": column,
                "unknown_count": count,
                "unknown_percentage": percentage
            })

unknown_df = pd.DataFrame(unknown_results)

if not unknown_df.empty:
    print(
        unknown_df
        .sort_values(
            "unknown_percentage",
            ascending=False
        )
        .round(2)
        .to_string(index=False)
    )
else:
    print("No unknown values found.")


# ============================================================
# 14. BUSINESS INSIGHTS
# ============================================================

print("\n13. KEY BUSINESS INSIGHTS")
print("-" * 75)

insights = [
    (
        f"The overall subscription rate is "
        f"{overall_rate:.2f}%, indicating that the campaign "
        f"has a relatively small positive-response segment."
    ),

    (
        f"The highest subscription-rate job category is "
        f"{job_rate.index[0]}, suggesting that customer occupation "
        f"may be useful for campaign targeting."
    ),

    (
        f"The highest-performing contact method is "
        f"{contact_rate.index[0]}, indicating that contact channel "
        f"may influence campaign response."
    ),

    (
        f"The strongest previous-campaign group is "
        f"{previous_rate.index[0]}, showing that previous campaign "
        f"history may provide useful predictive information."
    ),

    (
        f"The highest subscription-rate month is "
        f"{month_rate.index[0]}, suggesting that campaign timing "
        f"may be relevant."
    ),

    (
        "The number of campaign contacts should be examined carefully "
        "because repeated contacts may not always improve response."
    ),

    (
        "Economic indicators should be evaluated together with "
        "customer and campaign characteristics rather than interpreted "
        "as isolated business rules."
    )
]


for number, insight in enumerate(insights, start=1):
    print(f"{number}. {insight}")


# ============================================================
# 15. MODELING IMPLICATIONS
# ============================================================

print("\n14. MODELING IMPLICATIONS")
print("-" * 75)

modeling_points = [
    "The target variable is binary: yes/no.",
    "The dataset contains both numerical and categorical features.",
    "Class imbalance makes accuracy alone insufficient.",
    "Precision, recall, F1-score and ROC-AUC should be considered.",
    "Categorical features require encoding.",
    "Numerical features require appropriate preprocessing.",
    "duration must be excluded because it is unavailable when "
    "deciding which customers to target.",
    "The model should support customer-level campaign prioritization."
]

for number, point in enumerate(modeling_points, start=1):
    print(f"{number}. {point}")


# ============================================================
# 16. SAVE BUSINESS SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Subscribers",
        "Non-Subscribers",
        "Overall Subscription Rate",
        "Best Job Category",
        "Best Contact Method",
        "Best Month",
        "Best Previous Campaign Outcome"
    ],
    "Value": [
        len(df),
        int((df["target"] == 1).sum()),
        int((df["target"] == 0).sum()),
        f"{overall_rate:.2f}%",
        job_rate.index[0],
        contact_rate.index[0],
        month_rate.index[0],
        previous_rate.index[0]
    ]
})


summary.to_csv(
    OUTPUT_DIR / "eda_business_summary.csv",
    index=False
)


# ============================================================
# 17. SAVE ECONOMIC CORRELATIONS
# ============================================================

economic_corr.to_csv(
    OUTPUT_DIR / "economic_correlations.csv"
)


# ============================================================
# 18. SAVE INSIGHTS
# ============================================================

with open(
    OUTPUT_DIR / "key_business_insights.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BANK MARKETING PROJECT - KEY BUSINESS INSIGHTS\n"
    )

    file.write("=" * 70 + "\n\n")

    for number, insight in enumerate(insights, start=1):
        file.write(
            f"{number}. {insight}\n\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("EDA ANALYSIS & BUSINESS SUMMARY COMPLETE")
print("=" * 75)

print("\nFiles created:")
print("- eda_business_summary.csv")
print("- economic_correlations.csv")
print("- key_business_insights.txt")

print("\nAll files are saved inside:")
print(OUTPUT_DIR)
