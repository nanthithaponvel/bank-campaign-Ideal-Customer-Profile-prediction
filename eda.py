import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# LOAD DATASET
# ============================================================

file_path = "data/bank-additional-full.csv"

df = pd.read_csv(file_path, sep=";")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ============================================================
# 1. BASIC DATASET INFORMATION
# ============================================================

print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. DATA TYPES")
print(df.dtypes)

print("\n3. MISSING VALUES")
print(df.isnull().sum())

print("\n4. DUPLICATE ROWS")
print("Duplicates:", df.duplicated().sum())

# ============================================================
# 2. TARGET DISTRIBUTION
# ============================================================

print("\n5. TARGET DISTRIBUTION")
print(df["y"].value_counts())

print("\nTarget percentage:")
print(df["y"].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="y")
plt.title("Term Deposit Subscription Distribution")
plt.xlabel("Subscription")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("eda_outputs/target_distribution.png")
plt.show()

# ============================================================
# 3. AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=30, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("eda_outputs/age_distribution.png")
plt.show()

# ============================================================
# 4. SUBSCRIPTION RATE BY JOB
# ============================================================

job_rate = df.groupby("job")["y"].apply(
    lambda x: (x == "yes").mean() * 100
).sort_values(ascending=False)

print("\n6. SUBSCRIPTION RATE BY JOB")
print(job_rate)

plt.figure(figsize=(10, 5))
job_rate.plot(kind="bar")
plt.title("Subscription Rate by Job")
plt.xlabel("Job")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/subscription_by_job.png")
plt.show()

# ============================================================
# 5. SUBSCRIPTION RATE BY EDUCATION
# ============================================================

education_rate = df.groupby("education")["y"].apply(
    lambda x: (x == "yes").mean() * 100
).sort_values(ascending=False)

print("\n7. SUBSCRIPTION RATE BY EDUCATION")
print(education_rate)

plt.figure(figsize=(8, 5))
education_rate.plot(kind="bar")
plt.title("Subscription Rate by Education")
plt.xlabel("Education")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/subscription_by_education.png")
plt.show()

# ============================================================
# 6. SUBSCRIPTION RATE BY MARITAL STATUS
# ============================================================

marital_rate = df.groupby("marital")["y"].apply(
    lambda x: (x == "yes").mean() * 100
).sort_values(ascending=False)

print("\n8. SUBSCRIPTION RATE BY MARITAL STATUS")
print(marital_rate)

plt.figure(figsize=(7, 5))
marital_rate.plot(kind="bar")
plt.title("Subscription Rate by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_outputs/subscription_by_marital.png")
plt.show()

# ============================================================
# 7. SUBSCRIPTION RATE BY HOUSING LOAN
# ============================================================

housing_rate = df.groupby("housing")["y"].apply(
    lambda x: (x == "yes").mean() * 100
).sort_values(ascending=False)

print("\n9. SUBSCRIPTION RATE BY HOUSING LOAN")
print(housing_rate)

plt.figure(figsize=(7, 5))
housing_rate.plot(kind="bar")
plt.title("Subscription Rate by Housing Loan")
plt.xlabel("Housing Loan")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_outputs/subscription_by_housing.png")
plt.show()

# ============================================================
# 8. SUBSCRIPTION RATE BY PERSONAL LOAN
# ============================================================

loan_rate = df.groupby("loan")["y"].apply(
    lambda x: (x == "yes").mean() * 100
).sort_values(ascending=False)

print("\n10. SUBSCRIPTION RATE BY PERSONAL LOAN")
print(loan_rate)

plt.figure(figsize=(7, 5))
loan_rate.plot(kind="bar")
plt.title("Subscription Rate by Personal Loan")
plt.xlabel("Personal Loan")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_outputs/subscription_by_personal_loan.png")
plt.show()

# ============================================================
# 9. SUBSCRIPTION RATE BY CONTACT METHOD
# ============================================================

contact_rate = df.groupby("contact")["y"].apply(
    lambda x: (x == "yes").mean() * 100
).sort_values(ascending=False)

print("\n11. SUBSCRIPTION RATE BY CONTACT METHOD")
print(contact_rate)

plt.figure(figsize=(7, 5))
contact_rate.plot(kind="bar")
plt.title("Subscription Rate by Contact Method")
plt.xlabel("Contact Method")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_outputs/subscription_by_contact.png")
plt.show()

# ============================================================
# 10. SUBSCRIPTION RATE BY MONTH
# ============================================================

month_rate = df.groupby("month")["y"].apply(
    lambda x: (x == "yes").mean() * 100
)

print("\n12. SUBSCRIPTION RATE BY MONTH")
print(month_rate)

plt.figure(figsize=(9, 5))
month_rate.plot(kind="bar")
plt.title("Subscription Rate by Month")
plt.xlabel("Month")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_outputs/subscription_by_month.png")
plt.show()

# ============================================================
# 11. PREVIOUS CAMPAIGN OUTCOME
# ============================================================

poutcome_rate = df.groupby("poutcome")["y"].apply(
    lambda x: (x == "yes").mean() * 100
).sort_values(ascending=False)

print("\n13. SUBSCRIPTION RATE BY PREVIOUS CAMPAIGN OUTCOME")
print(poutcome_rate)

plt.figure(figsize=(7, 5))
poutcome_rate.plot(kind="bar")
plt.title("Subscription Rate by Previous Campaign Outcome")
plt.xlabel("Previous Campaign Outcome")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_outputs/subscription_by_poutcome.png")
plt.show()

# ============================================================
# 12. CAMPAIGN CONTACTS VS SUBSCRIPTION
# ============================================================

campaign_rate = df.groupby("campaign")["y"].apply(
    lambda x: (x == "yes").mean() * 100
)

plt.figure(figsize=(10, 5))
campaign_rate.plot()
plt.title("Subscription Rate vs Number of Campaign Contacts")
plt.xlabel("Number of Contacts During Campaign")
plt.ylabel("Subscription Rate (%)")
plt.tight_layout()
plt.savefig("eda_outputs/campaign_vs_subscription.png")
plt.show()

# ============================================================
# 13. NUMERICAL CORRELATION HEATMAP
# ============================================================

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("eda_outputs/correlation_heatmap.png")
plt.show()

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("Charts saved inside the eda_outputs folder.")
print("=" * 60)
