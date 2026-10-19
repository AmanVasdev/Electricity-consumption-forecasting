import pandas as pd
import numpy as np
import json
import os

df = pd.read_csv(r"project/data/processed/academic_features.csv")

print("==================================================")
print("1. DATASET SHAPE")
print("==================================================")
print(f"Shape: {df.shape} ({df.shape[0]:,} rows, {df.shape[1]} columns)")

print("\n==================================================")
print("2. ALL COLUMN NAMES (52 Total)")
print("==================================================")
for idx, col in enumerate(df.columns, 1):
    print(f"{idx:2d}. {col}")

print("\n==================================================")
print("3. FIRST 5 ROWS")
print("==================================================")
sample_cols = [
    "timestamp", "power", "energy_kwh", "power_lag_1", "power_lag_24",
    "power_rolling_mean_24", "power_rolling_std_24", "hour_sin", "is_weekend"
]
print(df[sample_cols].head().to_string())

print("\n==================================================")
print("4. LAST 5 ROWS")
print("==================================================")
print(df[sample_cols].tail().to_string())

print("\n==================================================")
print("5. TARGET MISSING COUNTS")
print("==================================================")
p_na = df["power"].isna().sum()
e_na = df["energy_kwh"].isna().sum()
print(f"power missing count:      {p_na:,} ({p_na/len(df)*100:.2f}%)")
print(f"energy_kwh missing count: {e_na:,} ({e_na/len(df)*100:.2f}%)")

print("\n==================================================")
print("6. TOP MISSING VALUE COUNTS IN ENGINEERED FEATURES")
print("==================================================")
missing_counts = df.isna().sum()
for col, cnt in missing_counts.sort_values(ascending=False).head(15).items():
    print(f"{col:30s}: {cnt:5d} ({cnt/len(df)*100:.2f}%)")

print("\n==================================================")
print("7. DATE RANGE & INTEGRITY")
print("==================================================")
dt = pd.to_datetime(df["timestamp"])
print(f"Start Date: {df['timestamp'].iloc[0]}")
print(f"End Date:   {df['timestamp'].iloc[-1]}")
print(f"Is strictly monotonic increasing: {dt.is_monotonic_increasing}")
print(f"Duplicate timestamps count:       {dt.duplicated().sum()}")

print("\n==================================================")
print("8. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
print("==================================================")
with open(r"project/reports/split_specification.json") as f:
    splits = json.load(f)["splits"]

for s_name, s_info in splits.items():
    print(f"{s_name.upper():10s}: {s_info['num_rows']:5d} rows ({s_info['percentage']}%) | {s_info['start_timestamp']} to {s_info['end_timestamp']}")

print("\n==================================================")
print("9. TARGET LEAKAGE CHECK (CURRENT / FUTURE TARGET)")
print("==================================================")
lag1_match = np.allclose(df["power_lag_1"].dropna().values, df["power"].shift(1).dropna().values)
lag24_match = np.allclose(df["power_lag_24"].dropna().values, df["power"].shift(24).dropna().values)
print(f"power_lag_1 strictly equals power.shift(1):   {lag1_match}")
print(f"power_lag_24 strictly equals power.shift(24): {lag24_match}")

# Check rolling mean independence
test_idx = 2000
orig_roll = df.loc[test_idx, "power_rolling_mean_24"]
s_test = df["power"].copy()
s_test.iloc[test_idx] = 999999999.0
new_roll = s_test.shift(1).rolling(24).mean().iloc[test_idx]
print(f"Rolling mean independent of current power:    {orig_roll == new_roll} (orig={orig_roll:.2f}, new={new_roll:.2f})")

print("\n==================================================")
print("10. GENERATED DELIVERABLES")
print("==================================================")
for p in [
    r"project/data/processed/academic_features.csv",
    r"project/src/feature_engineering.py",
    r"project/reports/split_specification.json",
    r"project/reports/feature_engineering_report.md"
]:
    print(f"{p}: {os.path.getsize(p):,} bytes")
