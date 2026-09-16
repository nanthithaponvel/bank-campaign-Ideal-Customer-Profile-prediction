import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# BANK MARKETING PROJECT
# BUSINESS PREDICTION ANALYSIS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

PREDICTION_FILE = (
    PROJECT_DIR
    / "models"
    / "customer_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "prediction_analysis"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


print("=" * 75)
print("BANK MARKETING PROJECT - BUSINESS PREDICTION ANALYSIS")
print("=" * 75)


# ============================================================
# 1. LOAD CUSTOMER PREDICTIONS
# ============================================================

if not PREDICTION_FILE.exists():

    raise FileNotFoundError(
        f"Prediction file not found:\n"
        f"{PREDICTION_FILE}\n\n"
        "Run prediction.py first."
    )


df = pd.read_csv(
    PREDICTION_FILE
)


print("\n1. PREDICTION DATA LOADED")
print("-" * 75)

print(
    f"Customers available: {len(df)}"
)


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "subscription_probability",
    "predicted_subscription",
    "campaign_priority"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        f"{missing_columns}"
    )


print(
    "[OK] Required prediction columns available."
)


# ============================================================
# 3. OVERALL PREDICTION SUMMARY
# ============================================================

total_customers = len(df)

average_probability = (
    df[
        "subscription_probability"
    ].mean()
)


predicted_yes = (
    df[
        "predicted_subscription"
    ]
    .eq("yes")
    .sum()
)


predicted_no = (
    df[
        "predicted_subscription"
    ]
    .eq("no")
    .sum()
)


print("\n2. OVERALL PREDICTION SUMMARY")
print("-" * 75)

print(
    f"Total customers scored : {total_customers}"
)

print(
    f"Average probability    : "
    f"{average_probability:.2%}"
)

print(
    f"Predicted subscribers  : {predicted_yes}"
)

print(
    f"Predicted non-subscribers: {predicted_no}"
)


# ============================================================
# 4. PRIORITY SUMMARY
# ============================================================

priority_order = [
    "High",
    "Medium",
    "Low"
]


priority_summary = (
    df[
        "campaign_priority"
    ]
    .value_counts()
    .reindex(
        priority_order,
        fill_value=0
    )
    .reset_index()
)


priority_summary.columns = [
    "Priority",
    "Customers"
]


priority_summary[
    "Percentage"
] = (
    priority_summary[
        "Customers"
    ]
    / total_customers
    * 100
)


print("\n3. CAMPAIGN PRIORITY SUMMARY")
print("-" * 75)

print(
    priority_summary.to_string(
        index=False
    )
)


priority_summary.to_csv(
    OUTPUT_DIR
    / "priority_summary.csv",
    index=False
)


# ============================================================
# 5. TOP 100 CUSTOMERS
# ============================================================

top_customers = (
    df
    .sort_values(
        "subscription_probability",
        ascending=False
    )
    .head(100)
    .copy()
)


top_customers[
    "subscription_probability"
] = (
    top_customers[
        "subscription_probability"
    ].round(4)
)


top_customers.to_csv(
    OUTPUT_DIR
    / "top_100_customers.csv",
    index=False
)


print("\n4. TOP CUSTOMER ANALYSIS")
print("-" * 75)

print(
    f"Top customer probability: "
    f"{df['subscription_probability'].max():.2%}"
)

print(
    f"Average probability of top 100: "
    f"{top_customers['subscription_probability'].mean():.2%}"
)


# ============================================================
# 6. JOB ANALYSIS
# ============================================================

if "job" in df.columns:

    job_analysis = (
        df
        .groupby("job")
        .agg(
            Customers=(
                "subscription_probability",
                "size"
            ),
            Average_Probability=(
                "subscription_probability",
                "mean"
            ),
            High_Priority=(
                "campaign_priority",
                lambda x: (x == "High").sum()
            ),
            Predicted_Subscriptions=(
                "predicted_subscription",
                lambda x: (x == "yes").sum()
            )
        )
        .reset_index()
    )


    job_analysis[
        "Average_Probability"
    ] = (
        job_analysis[
            "Average_Probability"
        ] * 100
    ).round(2)


    job_analysis = (
        job_analysis
        .sort_values(
            "Average_Probability",
            ascending=False
        )
    )


    job_analysis.to_csv(
        OUTPUT_DIR
        / "job_prediction_analysis.csv",
        index=False
    )


    print("\n5. JOB PREDICTION ANALYSIS")
    print("-" * 75)

    print(
        job_analysis.to_string(
            index=False
        )
    )


# ============================================================
# 7. CONTACT METHOD ANALYSIS
# ============================================================

if "contact" in df.columns:

    contact_analysis = (
        df
        .groupby("contact")
        .agg(
            Customers=(
                "subscription_probability",
                "size"
            ),
            Average_Probability=(
                "subscription_probability",
                "mean"
            ),
            High_Priority=(
                "campaign_priority",
                lambda x: (x == "High").sum()
            ),
            Predicted_Subscriptions=(
                "predicted_subscription",
                lambda x: (x == "yes").sum()
            )
        )
        .reset_index()
    )


    contact_analysis[
        "Average_Probability"
    ] = (
        contact_analysis[
            "Average_Probability"
        ] * 100
    ).round(2)


    contact_analysis = (
        contact_analysis
        .sort_values(
            "Average_Probability",
            ascending=False
        )
    )


    contact_analysis.to_csv(
        OUTPUT_DIR
        / "contact_prediction_analysis.csv",
        index=False
    )


    print("\n6. CONTACT METHOD ANALYSIS")
    print("-" * 75)

    print(
        contact_analysis.to_string(
            index=False
        )
    )


# ============================================================
# 8. MONTH ANALYSIS
# ============================================================

if "month" in df.columns:

    month_analysis = (
        df
        .groupby("month")
        .agg(
            Customers=(
                "subscription_probability",
                "size"
            ),
            Average_Probability=(
                "subscription_probability",
                "mean"
            ),
            High_Priority=(
                "campaign_priority",
                lambda x: (x == "High").sum()
            ),
            Predicted_Subscriptions=(
                "predicted_subscription",
                lambda x: (x == "yes").sum()
            )
        )
        .reset_index()
    )


    month_analysis[
        "Average_Probability"
    ] = (
        month_analysis[
            "Average_Probability"
        ] * 100
    ).round(2)


    month_analysis = (
        month_analysis
        .sort_values(
            "Average_Probability",
            ascending=False
        )
    )


    month_analysis.to_csv(
        OUTPUT_DIR
        / "month_prediction_analysis.csv",
        index=False
    )


    print("\n7. MONTH PREDICTION ANALYSIS")
    print("-" * 75)

    print(
        month_analysis.to_string(
            index=False
        )
    )


# ============================================================
# 9. PREVIOUS CAMPAIGN OUTCOME ANALYSIS
# ============================================================

if "poutcome" in df.columns:

    poutcome_analysis = (
        df
        .groupby("poutcome")
        .agg(
            Customers=(
                "subscription_probability",
                "size"
            ),
            Average_Probability=(
                "subscription_probability",
                "mean"
            ),
            High_Priority=(
                "campaign_priority",
                lambda x: (x == "High").sum()
            ),
            Predicted_Subscriptions=(
                "predicted_subscription",
                lambda x: (x == "yes").sum()
            )
        )
        .reset_index()
    )


    poutcome_analysis[
        "Average_Probability"
    ] = (
        poutcome_analysis[
            "Average_Probability"
        ] * 100
    ).round(2)


    poutcome_analysis = (
        poutcome_analysis
        .sort_values(
            "Average_Probability",
            ascending=False
        )
    )


    poutcome_analysis.to_csv(
        OUTPUT_DIR
        / "previous_campaign_analysis.csv",
        index=False
    )


    print("\n8. PREVIOUS CAMPAIGN OUTCOME ANALYSIS")
    print("-" * 75)

    print(
        poutcome_analysis.to_string(
            index=False
        )
    )


# ============================================================
# 10. CAMPAIGN CONTACT FREQUENCY
# ============================================================

if "campaign" in df.columns:

    campaign_analysis = (
        df
        .groupby("campaign")
        .agg(
            Customers=(
                "subscription_probability",
                "size"
            ),
            Average_Probability=(
                "subscription_probability",
                "mean"
            ),
            High_Priority=(
                "campaign_priority",
                lambda x: (x == "High").sum()
            ),
            Predicted_Subscriptions=(
                "predicted_subscription",
                lambda x: (x == "yes").sum()
            )
        )
        .reset_index()
    )


    campaign_analysis[
        "Average_Probability"
    ] = (
        campaign_analysis[
            "Average_Probability"
        ] * 100
    ).round(2)


    campaign_analysis = (
        campaign_analysis
        .sort_values(
            "campaign"
        )
    )


    campaign_analysis.to_csv(
        OUTPUT_DIR
        / "campaign_prediction_analysis.csv",
        index=False
    )


    print("\n9. CAMPAIGN CONTACT ANALYSIS")
    print("-" * 75)

    print(
        campaign_analysis.to_string(
            index=False
        )
    )


# ============================================================
# 11. TOP 10% CUSTOMER ANALYSIS
# ============================================================

top_10_count = max(
    1,
    int(
        np.ceil(
            total_customers * 0.10
        )
    )
)


top_10 = (
    df
    .sort_values(
        "subscription_probability",
        ascending=False
    )
    .head(
        top_10_count
    )
)


top_10_avg_probability = (
    top_10[
        "subscription_probability"
    ].mean()
)


top_10_predicted_yes = (
    top_10[
        "predicted_subscription"
    ]
    .eq("yes")
    .sum()
)


top_10_analysis = pd.DataFrame({

    "Metric": [
        "Total Customers",
        "Top 10% Customers",
        "Top 10% Average Probability",
        "Top 10% Predicted Subscribers"
    ],

    "Value": [
        total_customers,
        top_10_count,
        round(
            top_10_avg_probability,
            4
        ),
        top_10_predicted_yes
    ]
})


top_10_analysis.to_csv(
    OUTPUT_DIR
    / "top_10_percent_analysis.csv",
    index=False
)


print("\n10. TOP 10% CUSTOMER ANALYSIS")
print("-" * 75)

print(
    f"Top 10% customers: "
    f"{top_10_count}"
)

print(
    f"Average probability: "
    f"{top_10_avg_probability:.2%}"
)

print(
    f"Predicted subscribers: "
    f"{top_10_predicted_yes}"
)


# ============================================================
# 12. TOP 20% CUSTOMER ANALYSIS
# ============================================================

top_20_count = max(
    1,
    int(
        np.ceil(
            total_customers * 0.20
        )
    )
)


top_20 = (
    df
    .sort_values(
        "subscription_probability",
        ascending=False
    )
    .head(
        top_20_count
    )
)


top_20_avg_probability = (
    top_20[
        "subscription_probability"
    ].mean()
)


top_20_predicted_yes = (
    top_20[
        "predicted_subscription"
    ]
    .eq("yes")
    .sum()
)


print("\n11. TOP 20% CUSTOMER ANALYSIS")
print("-" * 75)

print(
    f"Top 20% customers: "
    f"{top_20_count}"
)

print(
    f"Average probability: "
    f"{top_20_avg_probability:.2%}"
)

print(
    f"Predicted subscribers: "
    f"{top_20_predicted_yes}"
)


# ============================================================
# 13. TOP 30% CUSTOMER ANALYSIS
# ============================================================

top_30_count = max(
    1,
    int(
        np.ceil(
            total_customers * 0.30
        )
    )
)


top_30 = (
    df
    .sort_values(
        "subscription_probability",
        ascending=False
    )
    .head(
        top_30_count
    )
)


top_30_avg_probability = (
    top_30[
        "subscription_probability"
    ].mean()
)


top_30_predicted_yes = (
    top_30[
        "predicted_subscription"
    ]
    .eq("yes")
    .sum()
)


print("\n12. TOP 30% CUSTOMER ANALYSIS")
print("-" * 75)

print(
    f"Top 30% customers: "
    f"{top_30_count}"
)

print(
    f"Average probability: "
    f"{top_30_avg_probability:.2%}"
)

print(
    f"Predicted subscribers: "
    f"{top_30_predicted_yes}"
)


# ============================================================
# 14. BUSINESS SUMMARY
# ============================================================

summary_lines = [

    "BANK MARKETING PROJECT",
    "BUSINESS PREDICTION ANALYSIS",
    "=" * 60,
    "",
    f"Total customers scored: {total_customers}",
    f"Average predicted subscription probability: "
    f"{average_probability:.2%}",
    f"Predicted subscribers: {predicted_yes}",
    f"Predicted non-subscribers: {predicted_no}",
    "",
    "CAMPAIGN PRIORITY",
    "-" * 30,
    f"High priority: "
    f"{priority_summary.loc[0, 'Customers']}",
    f"Medium priority: "
    f"{priority_summary.loc[1, 'Customers']}",
    f"Low priority: "
    f"{priority_summary.loc[2, 'Customers']}",
    "",
    "TOP 10%",
    "-" * 30,
    f"Customers: {top_10_count}",
    f"Average probability: "
    f"{top_10_avg_probability:.2%}",
    f"Predicted subscribers: "
    f"{top_10_predicted_yes}",
    "",
    "TOP 20%",
    "-" * 30,
    f"Customers: {top_20_count}",
    f"Average probability: "
    f"{top_20_avg_probability:.2%}",
    f"Predicted subscribers: "
    f"{top_20_predicted_yes}",
    "",
    "TOP 30%",
    "-" * 30,
    f"Customers: {top_30_count}",
    f"Average probability: "
    f"{top_30_avg_probability:.2%}",
    f"Predicted subscribers: "
    f"{top_30_predicted_yes}",
    "",
    "BUSINESS INTERPRETATION",
    "-" * 30,
    "The model provides a customer-level ranking",
    "based on predicted subscription probability.",
    "",
    "Higher-probability customers can be prioritized",
    "for future marketing campaigns.",
    "",
    "The ranking can help the bank allocate campaign",
    "resources more efficiently than contacting all",
    "customers equally.",
    "",
    "This analysis is predictive and should not be",
    "interpreted as a guarantee that a customer will",
    "subscribe."
]


summary_file = (
    OUTPUT_DIR
    / "business_prediction_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary_lines)
    )


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)

print(
    "BUSINESS PREDICTION ANALYSIS COMPLETE"
)

print("=" * 75)

print("\nGenerated files:")

print(
    "- prediction_analysis/priority_summary.csv"
)

print(
    "- prediction_analysis/top_100_customers.csv"
)

print(
    "- prediction_analysis/job_prediction_analysis.csv"
)

print(
    "- prediction_analysis/contact_prediction_analysis.csv"
)

print(
    "- prediction_analysis/month_prediction_analysis.csv"
)

print(
    "- prediction_analysis/previous_campaign_analysis.csv"
)

print(
    "- prediction_analysis/campaign_prediction_analysis.csv"
)

print(
    "- prediction_analysis/top_10_percent_analysis.csv"
)

print(
    "- prediction_analysis/business_prediction_summary.txt"
)


print("\nNext stage:")

print(
    "Prepare the final dashboard."
)

print(
    "\n" + "=" * 75
)