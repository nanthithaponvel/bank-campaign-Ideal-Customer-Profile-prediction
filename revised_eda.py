import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# BANK MARKETING PROJECT - REVISED EDA
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
        f"Dataset not found:\n{DATA_FILE}\n\n"
        "Run ingest.py first."
    )

df = pd.read_csv(DATA_FILE, sep=";")


print("=" * 70)
print("BANK MARKETING PROJECT - REVISED EDA")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. CREATE BINARY TARGET
# ============================================================

df["target"] = df["y"].map({
    "no": 0,
    "yes": 1
})


# ============================================================
# 3. OVERALL SUBSCRIPTION RATE
# ============================================================

subscription_rate = df["target"].mean() * 100

print("\n1. OVERALL SUBSCRIPTION RATE")
print(f"Subscription rate: {subscription_rate:.2f}%")


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\n2. TARGET DISTRIBUTION")

print(df["y"].value_counts())

plt.figure(figsize=(7, 5))

df["y"].value_counts().plot(kind="bar")

plt.title("Term Deposit Subscription Distribution")
plt.xlabel("Subscription")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_target_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. SUBSCRIPTION RATE BY JOB
# ============================================================

print("\n3. SUBSCRIPTION RATE BY JOB")

job_rate = (
    df.groupby("job")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(job_rate.round(2))

plt.figure(figsize=(10, 6))

job_rate.plot(kind="bar")

plt.title("Term Deposit Subscription Rate by Job")
plt.xlabel("Job")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_subscription_by_job.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. SUBSCRIPTION RATE BY EDUCATION
# ============================================================

print("\n4. SUBSCRIPTION RATE BY EDUCATION")

education_rate = (
    df.groupby("education")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(education_rate.round(2))

plt.figure(figsize=(10, 6))

education_rate.plot(kind="bar")

plt.title("Term Deposit Subscription Rate by Education")
plt.xlabel("Education")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_subscription_by_education.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. SUBSCRIPTION RATE BY CONTACT METHOD
# ============================================================

print("\n5. SUBSCRIPTION RATE BY CONTACT METHOD")

contact_rate = (
    df.groupby("contact")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(contact_rate.round(2))

plt.figure(figsize=(7, 5))

contact_rate.plot(kind="bar")

plt.title("Term Deposit Subscription Rate by Contact Method")
plt.xlabel("Contact Method")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_subscription_by_contact.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. SUBSCRIPTION RATE BY MONTH
# ============================================================

print("\n6. SUBSCRIPTION RATE BY MONTH")

month_rate = (
    df.groupby("month")["target"]
    .mean()
    .mul(100)
)

print(month_rate.round(2))

plt.figure(figsize=(9, 5))

month_rate.plot(kind="bar")

plt.title("Term Deposit Subscription Rate by Month")
plt.xlabel("Month")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_subscription_by_month.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. SUBSCRIPTION RATE BY PREVIOUS CAMPAIGN OUTCOME
# ============================================================

print("\n7. SUBSCRIPTION RATE BY PREVIOUS CAMPAIGN OUTCOME")

previous_outcome_rate = (
    df.groupby("poutcome")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(previous_outcome_rate.round(2))

plt.figure(figsize=(8, 5))

previous_outcome_rate.plot(kind="bar")

plt.title("Subscription Rate by Previous Campaign Outcome")
plt.xlabel("Previous Campaign Outcome")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_subscription_by_previous_outcome.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. SUBSCRIPTION RATE BY NUMBER OF CONTACTS
# ============================================================

print("\n8. SUBSCRIPTION RATE BY NUMBER OF CONTACTS")

campaign_rate = (
    df.groupby("campaign")["target"]
    .mean()
    .mul(100)
)

print(campaign_rate.head(15).round(2))

plt.figure(figsize=(10, 5))

campaign_rate.head(15).plot(kind="bar")

plt.title("Subscription Rate by Number of Campaign Contacts")
plt.xlabel("Number of Contacts")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_subscription_by_campaign_contacts.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. AGE DISTRIBUTION
# ============================================================

print("\n9. AGE ANALYSIS")

print(df["age"].describe().round(2))

plt.figure(figsize=(8, 5))

df["age"].plot(
    kind="hist",
    bins=30
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_age_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. AGE VS SUBSCRIPTION
# ============================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="age",
    by="y"
)

plt.title("Age Distribution by Term Deposit Subscription")
plt.suptitle("")
plt.xlabel("Subscription")
plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "09_age_vs_subscription.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. HOUSING LOAN
# ============================================================

print("\n10. SUBSCRIPTION RATE BY HOUSING LOAN")

housing_rate = (
    df.groupby("housing")["target"]
    .mean()
    .mul(100)
)

print(housing_rate.round(2))

plt.figure(figsize=(7, 5))

housing_rate.plot(kind="bar")

plt.title("Subscription Rate by Housing Loan")
plt.xlabel("Housing Loan")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "10_subscription_by_housing.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. PERSONAL LOAN
# ============================================================

print("\n11. SUBSCRIPTION RATE BY PERSONAL LOAN")

loan_rate = (
    df.groupby("loan")["target"]
    .mean()
    .mul(100)
)

print(loan_rate.round(2))

plt.figure(figsize=(7, 5))

loan_rate.plot(kind="bar")

plt.title("Subscription Rate by Personal Loan")
plt.xlabel("Personal Loan")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "11_subscription_by_personal_loan.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. ECONOMIC FEATURES
# ============================================================

economic_features = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

print("\n12. ECONOMIC FEATURE CORRELATIONS")

economic_correlation = (
    df[economic_features + ["target"]]
    .corr()["target"]
    .sort_values(ascending=False)
)

print(economic_correlation.round(3))


# ============================================================
# 16. NUMERICAL CORRELATION
# ============================================================

print("\n13. NUMERICAL FEATURE CORRELATIONS")

numeric_columns = [
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

correlation = (
    df[numeric_columns + ["target"]]
    .corr()["target"]
    .sort_values(ascending=False)
)

print(correlation.round(3))


# ============================================================
# 17. SAVE SUMMARY TABLES
# ============================================================

job_rate.to_csv(
    OUTPUT_DIR / "job_subscription_rate.csv"
)

education_rate.to_csv(
    OUTPUT_DIR / "education_subscription_rate.csv"
)

contact_rate.to_csv(
    OUTPUT_DIR / "contact_subscription_rate.csv"
)

month_rate.to_csv(
    OUTPUT_DIR / "month_subscription_rate.csv"
)

previous_outcome_rate.to_csv(
    OUTPUT_DIR / "previous_outcome_subscription_rate.csv"
)

housing_rate.to_csv(
    OUTPUT_DIR / "housing_subscription_rate.csv"
)

loan_rate.to_csv(
    OUTPUT_DIR / "loan_subscription_rate.csv"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("REVISED EDA COMPLETE")
print("=" * 70)

print("\nEDA outputs saved to:")
print(OUTPUT_DIR)