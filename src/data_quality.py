import pandas as pd

# 1. load data
df = pd.read_csv("data/raw/full_data.csv")

# 2. basic info
print("SHAPE:", df.shape)
print("\nCOLUMNS:", df.columns)

# 3. check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# 4. check datetime types
print("\nDTYPES:")
print(df.dtypes)

# 5. convert (safety check, not mutation yet)
df["started_at_test"] = pd.to_datetime(df["started_at"], errors="coerce")
df["finished_at_test"] = pd.to_datetime(df["finished_at"], errors="coerce")

print("\nINVALID DATES (started_at):", df["started_at_test"].isna().sum())
print("INVALID DATES (finished_at):", df["finished_at_test"].isna().sum())

# 6. logical check
invalid_duration = (df["finished_at_test"] < df["started_at_test"]).sum()
print("\nINVALID DURATIONS:", invalid_duration)

# 7. duplicates
print("\nDUPLICATES:", df.duplicated().sum())