import pandas as pd

# Path to the Bank Marketing dataset
file_path = "data/bank-additional-full.csv"

# Load the dataset
# The UCI Bank Marketing dataset uses semicolon (;) as the separator
df = pd.read_csv(file_path, sep=";")

print("=" * 60)
print("BANK MARKETING DATASET VERIFICATION")
print("=" * 60)

# 1. Dataset shape
print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column names
print("\n2. COLUMN NAMES")
for column in df.columns:
    print("-", column)

# 3. Data types
print("\n3. DATA TYPES")
print(df.dtypes)

# 4. Missing values
print("\n4. MISSING VALUES")
print(df.isnull().sum())

# 5. Duplicate rows
print("\n5. DUPLICATE ROWS")
print("Number of duplicate rows:", df.duplicated().sum())

# 6. Target variable
print("\n6. TARGET VARIABLE")
print("Target column:", "y")
print("\nTarget values:")
print(df["y"].value_counts())

# 7. First 5 rows
print("\n7. FIRST 5 ROWS")
print(df.head())

print("\n" + "=" * 60)
print("DATASET VERIFICATION COMPLETE")
print("=" * 60)