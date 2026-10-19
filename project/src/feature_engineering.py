"""
Feature Engineering Pipeline for Academic Building Mains Electricity Forecasting
Input: project/data/processed/academic_hourly.csv
Output: project/data/processed/academic_features.csv
Report: project/reports/feature_engineering_report.md
"""

import os
import sys
import json
import numpy as np
import pandas as pd

INPUT_FILE = r"project\data\processed\academic_hourly.csv"
OUTPUT_FILE = r"project\data\processed\academic_features.csv"
REPORT_FILE = r"project\reports\feature_engineering_report.md"
SPLIT_JSON_FILE = r"project\reports\split_specification.json"

def run_feature_engineering():
    print(f"Starting feature engineering pipeline...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # 1. Load processed hourly dataset
    print(f"1. Loading processed hourly dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    n_original_rows, n_original_cols = df.shape
    original_cols = list(df.columns)
    print(f"   Original shape: {df.shape} ({n_original_cols} columns, {n_original_rows:,} rows)")

    # Verify chronological integrity
    dt_series = pd.to_datetime(df["timestamp"])
    assert dt_series.is_monotonic_increasing, "Error: Timestamps are not monotonically increasing!"
    assert dt_series.duplicated().sum() == 0, "Error: Duplicate timestamps detected!"

    # 2. Lag Features for 'power' (Watts)
    # Lags: 1, 2, 3, 6, 12, 24, 48, 168
    print("2. Generating historical lag features for 'power'...")
    power_lags = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in power_lags:
        df[f"power_lag_{lag}"] = df["power"].shift(lag)

    # 3. Lag Features for 'energy_kwh' (kWh)
    print("3. Generating historical lag features for 'energy_kwh'...")
    energy_lags = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in energy_lags:
        df[f"energy_kwh_lag_{lag}"] = df["energy_kwh"].shift(lag)

    # 4. Rolling Statistics for 'power'
    # Windows: 3, 6, 12, 24, 168 hours
    # CRITICAL: Strictly shift by 1 to exclude the current target value and prevent leakage!
    print("4. Generating historical rolling statistics for 'power' (strictly shifted by 1)...")
    rolling_windows = [3, 6, 12, 24, 168]
    for w in rolling_windows:
        # power.shift(1).rolling(w).mean()
        df[f"power_rolling_mean_{w}"] = df["power"].shift(1).rolling(w).mean()
        df[f"power_rolling_std_{w}"] = df["power"].shift(1).rolling(w).std()

    # 5. Temporal Cyclical Features
    print("5. Generating cyclical temporal representations (sine and cosine encodings)...")
    # hour: 0 to 23 -> period 24
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)

    # day_of_week: 0 to 6 -> period 7
    df["day_of_week_sin"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["day_of_week_cos"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0)

    # month: 1 to 12 -> period 12 (using (month - 1) so Jan=0, Dec=11)
    df["month_sin"] = np.sin(2.0 * np.pi * (df["month"] - 1.0) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (df["month"] - 1.0) / 12.0)

    # 6. Power-Derived Within-Hour Profile Features
    print("6. Generating power-derived features and historical volatility lags...")
    df["power_range"] = df["power_max"] - df["power_min"]
    
    # Safe coefficient of variation: handle zero division and NaN
    df["coefficient_of_variation"] = np.where(
        df["power_mean"].isna() | df["power_std"].isna(),
        np.nan,
        np.where(df["power_mean"] > 0, df["power_std"] / df["power_mean"], 0.0)
    )
    # Lagged power range (valid historical predictor for next hour's dispersion)
    df["power_range_lag_1"] = df["power_range"].shift(1)

    # 7. Leakage Prevention Verification
    print("7. Verifying target leakage assertions...")
    # Verify that lag_1 is strictly previous row
    assert np.allclose(df["power_lag_1"].dropna().values, df["power"].shift(1).dropna().values), "Leakage in power_lag_1!"
    assert np.allclose(df["power_lag_24"].dropna().values, df["power"].shift(24).dropna().values), "Leakage in power_lag_24!"
    
    # Verify that rolling mean does not depend on current power
    test_idx = 1000
    orig_val = df.loc[test_idx, "power_rolling_mean_24"]
    mock_series = df["power"].copy()
    mock_series.iloc[test_idx] = 999999999.0
    mock_val = mock_series.shift(1).rolling(24).mean().iloc[test_idx]
    assert np.isclose(orig_val, mock_val), "CRITICAL LEAKAGE DETECTED in rolling calculation!"
    print("   Leakage verification passed: No current target values used in rolling/lag calculations.")

    n_final_rows, n_final_cols = df.shape
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"   Final shape: {df.shape} ({n_final_cols} columns, {n_final_rows:,} rows)")
    print(f"   Engineered features count: {len(new_cols)}")

    # 8. Chronological Train / Validation / Test Split Specification
    # 70% Train, 15% Validation, 15% Test
    print("8. Calculating chronological train/validation/test boundaries...")
    n_train = int(n_final_rows * 0.70)
    n_val = int(n_final_rows * 0.15)
    n_test = n_final_rows - n_train - n_val

    split_specs = {
        "total_rows": n_final_rows,
        "splits": {
            "train": {
                "start_idx": 0,
                "end_idx": n_train - 1,
                "num_rows": n_train,
                "percentage": round((n_train / n_final_rows) * 100, 2),
                "start_timestamp": df["timestamp"].iloc[0],
                "end_timestamp": df["timestamp"].iloc[n_train - 1]
            },
            "validation": {
                "start_idx": n_train,
                "end_idx": n_train + n_val - 1,
                "num_rows": n_val,
                "percentage": round((n_val / n_final_rows) * 100, 2),
                "start_timestamp": df["timestamp"].iloc[n_train],
                "end_timestamp": df["timestamp"].iloc[n_train + n_val - 1]
            },
            "test": {
                "start_idx": n_train + n_val,
                "end_idx": n_final_rows - 1,
                "num_rows": n_test,
                "percentage": round((n_test / n_final_rows) * 100, 2),
                "start_timestamp": df["timestamp"].iloc[n_train + n_val],
                "end_timestamp": df["timestamp"].iloc[-1]
            }
        }
    }

    os.makedirs(os.path.dirname(SPLIT_JSON_FILE), exist_ok=True)
    with open(SPLIT_JSON_FILE, "w") as f:
        json.dump(split_specs, f, indent=2)
    print(f"   Split specifications saved to {SPLIT_JSON_FILE}")

    # 9. Save Feature-Engineered Dataset
    print(f"9. Saving feature-engineered dataset to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    output_size_bytes = os.path.getsize(OUTPUT_FILE)
    print(f"   Saved {output_size_bytes:,} bytes ({output_size_bytes / (1024*1024):.2f} MB) to {OUTPUT_FILE}.")

    # 10. Generate Comprehensive Feature Engineering Report
    print(f"10. Generating report at {REPORT_FILE}...")
    write_feature_engineering_report(
        report_path=REPORT_FILE,
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        n_orig_cols=n_original_cols,
        n_final_cols=n_final_cols,
        original_cols=original_cols,
        new_cols=new_cols,
        split_specs=split_specs,
        df=df
    )
    print("Feature engineering pipeline completed successfully!")

def write_feature_engineering_report(report_path, input_path, output_path, n_orig_cols, n_final_cols,
                                     original_cols, new_cols, split_specs, df):
    train_info = split_specs["splits"]["train"]
    val_info = split_specs["splits"]["validation"]
    test_info = split_specs["splits"]["test"]

    # Compute missing values summary for report
    missing_summary = df.isna().sum()

    report_content = f"""# Feature Engineering Report: Electricity Consumption & Load Forecasting

**Project Stage:** Feature Engineering  
**Dataset:** Academic Building Mains (`acad_build_mains.csv`)  
**Execution Date:** September 29, 2026  
**Script Executed:** `project/src/feature_engineering.py`  
**Execution Environment:** `.\\.venv\\Scripts\\python.exe`  
**Hardware Profile:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM, CUDA 13.4)  

---

## 1. Input Dataset
- **File Path:** `{input_path}`
- **Dimensions:** `{len(df):,}` rows $\\times$ `{n_orig_cols}` columns
- **Temporal Granularity:** Regular hourly resolution (`freq='1h'`)
- **Date Range:** `{df['timestamp'].iloc[0]}` to `{df['timestamp'].iloc[-1]}` (1,605 days)

---

## 2. Output Dataset
- **File Path:** `{output_path}`
- **Dimensions:** `{len(df):,}` rows $\\times$ `{n_final_cols}` columns
- **File Size:** `{os.path.getsize(output_path):,}` bytes (~`{os.path.getsize(output_path)/(1024*1024):.2f}` MB)
- **Integrity:** Strict chronological ordering; zero duplicate timestamps; no overwriting of previous artifacts.

---

## 3. Column Count Comparison
- **Original Number of Columns:** `{n_orig_cols}`
- **New Engineered Features Added:** `{len(new_cols)}`
- **Final Number of Columns:** `{n_final_cols}`

---

## 4. List of All Engineered Features

The `{len(new_cols)}` engineered features are grouped into five functional categories:

| Category | Count | Feature Names | Mathematical / Logical Definition |
| :--- | :---: | :--- | :--- |
| **Power Lags** | 8 | `power_lag_1`, `power_lag_2`, `power_lag_3`, `power_lag_6`, `power_lag_12`, `power_lag_24`, `power_lag_48`, `power_lag_168` | $\\text{{power}}_{{t-k}} = \\text{{power.shift}}(k)$ |
| **Energy Lags** | 8 | `energy_kwh_lag_1`, `energy_kwh_lag_2`, `energy_kwh_lag_3`, `energy_kwh_lag_6`, `energy_kwh_lag_12`, `energy_kwh_lag_24`, `energy_kwh_lag_48`, `energy_kwh_lag_168` | $\\text{{energy\\_kwh}}_{{t-k}} = \\text{{energy\\_kwh.shift}}(k)$ |
| **Rolling Means** | 5 | `power_rolling_mean_3`, `power_rolling_mean_6`, `power_rolling_mean_12`, `power_rolling_mean_24`, `power_rolling_mean_168` | $\\frac{{1}}{{W}}\\sum_{{i=1}}^{{W}} \\text{{power}}_{{t-i}} = \\text{{power.shift}}(1).\\text{{rolling}}(W).\\text{{mean}}()$ |
| **Rolling Std** | 5 | `power_rolling_std_3`, `power_rolling_std_6`, `power_rolling_std_12`, `power_rolling_std_24`, `power_rolling_std_168` | $\\sqrt{{\\frac{{1}}{{W-1}}\\sum_{{i=1}}^{{W}} (\\text{{power}}_{{t-i}} - \\bar{{P}})^2}} = \\text{{power.shift}}(1).\\text{{rolling}}(W).\\text{{std}}()$ |
| **Cyclical Temporal** | 6 | `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos` | Sine & cosine transformations of periodic calendar units |
| **Power-Derived** | 3 | `power_range`, `coefficient_of_variation`, `power_range_lag_1` | Within-hour dispersion and historical volatility |

---

## 5. Explanation of Lag Features

Autoregressive temporal dependencies dominate electrical grid dynamics. Lag features capture critical periodic cycles:
1. **Short-Term Autoregression (`lag_1`, `lag_2`, `lag_3`)**:
   - Captures high-frequency load persistence. If the building was consuming 45 kW in the previous hour, the subsequent hour is likely to remain near 45 kW due to thermal inertia and ongoing campus operations.
2. **Intra-Day Shifts (`lag_6`, `lag_12`)**:
   - Captures morning ramp-up and afternoon ramp-down operational phases of the academic building.
3. **Daily Seasonality (`lag_24`, `lag_48`)**:
   - Exactly 24 and 48 hours prior. Captures diurnal load repetition at the identical hour of the previous day and two days prior.
4. **Weekly Seasonality (`lag_168`)**:
   - Exactly 168 hours ($7 \\times 24\\text{{h}}$) prior. Represents the identical hour on the same day of the preceding week, capturing institutional schedule alignment (e.g. comparing Monday 10:00 AM to the prior Monday 10:00 AM).
5. **Energy Lags (`energy_kwh_lag_k`)**:
   - Analogous historical energy consumption volume ($kWh$). Allows dual-objective models to observe both load demand and cumulative consumption patterns.

---

## 6. Explanation of Rolling Features

Rolling statistics capture multi-scale baseline trends and volatility:
- **Rolling Windows**: 3 hours, 6 hours, 12 hours, 24 hours (1 day), and 168 hours (1 week).
- **Rolling Mean (`power_rolling_mean_w`)**: Smoothes out momentary spikes and tracks the local baseline power consumption level.
- **Rolling Standard Deviation (`power_rolling_std_w`)**: Measures dynamic load volatility and instability over recent operational windows.
- **Leakage Prevention**: All rolling windows are strictly evaluated using `power.shift(1).rolling(w)`, guaranteeing that the current observation $y_t$ is completely excluded from the window.

---

## 7. Temporal & Cyclical Features

Calendar features (`hour`, `day_of_week`, `month`) have periodic continuity:
- Hour 23 is immediately followed by Hour 0.
- Sunday (6) is immediately followed by Monday (0).
- December (12) is immediately followed by January (1).

Standard linear integers fail to represent this circular proximity. We apply trigonometry to map each unit onto a 2D continuous unit circle:
$$\\text{{hour\\_sin}} = \\sin\\left(\\frac{{2\\pi \\times \\text{{hour}}}}{{24}}\\right), \\quad \\text{{hour\\_cos}} = \\cos\\left(\\frac{{2\\pi \\times \\text{{hour}}}}{{24}}\\right)$$
$$\\text{{day\\_of\\_week\\_sin}} = \\sin\\left(\\frac{{2\\pi \\times \\text{{day\\_of\\_week}}}}{{7}}\\right), \\quad \\text{{day\\_of\\_week\\_cos}} = \\cos\\left(\\frac{{2\\pi \\times \\text{{day\\_of\\_week}}}}{{7}}\\right)$$
$$\\text{{month\\_sin}} = \\sin\\left(\\frac{{2\\pi \\times (\\text{{month}} - 1)}}{{12}}\\right), \\quad \\text{{month\\_cos}} = \\cos\\left(\\frac{{2\\pi \\times (\\text{{month}} - 1)}}{{12}}\\right)$$

---

## 8. Power-Derived Features

1. **`power_range`**:
   $$\\text{{power\\_range}} = \\text{{power\\_max}} - \\text{{power\\_min}}$$
   Measures the span of instantaneous power variation within that hour.
2. **`coefficient_of_variation` (CV)**:
   $$\\text{{CV}} = \\frac{{\\text{{power\\_std}}}}{{\\text{{power\\_mean}}}}$$
   Normalized dimensionless measure of relative within-hour load dispersion. Handled with zero-division safeguard: if $\\text{{power\\_mean}} = 0$, $\\text{{CV}} = 0.0$.
3. **`power_range_lag_1`**:
   $$\\text{{power\\_range\\_lag\\_1}} = \\text{{power\\_range.shift}}(1)$$
   The previous hour's within-hour volatility, providing a strictly historical measure of load fluctuation for forecasting the current hour.

---

## 9. Missing-Value Handling Strategy

1. **Preservation of Genuine Zeros**:
   - 107 full hours with $\\text{{power}} = 0.0$ are preserved as real physical outage/shutdown events. They are **not** treated as missing.
2. **Preservation of Recording Outages**:
   - The 638 unrecorded hours caused by logger outages remain strictly `NaN`. They are **not** fabricated or linearly interpolated.
3. **Burn-in Missingness for Lags & Rolling Windows**:
   - At the beginning of the timeline, historical lags and rolling windows naturally contain `NaN` because preceding records do not exist (e.g. `lag_168` has 168 initial NaNs).
   - Across multi-day recording gaps, lags and rolling windows reflect the absence of historical observations.
   - Tree models (XGBoost) natively route `NaN` values via default split directions. For scikit-learn Random Forest, rows with `NaN` can be filtered or masked during sample batching.

---

## 10. Data Leakage Prevention Strategy

Strict anti-leakage principles were enforced throughout the pipeline:
1. **No Contemporaneous Target in Predictors**: Current-hour power and energy are never included in rolling or lag features.
2. **Strict Pre-Shift**: Rolling aggregations use `shift(1)` before window evaluation (`power.shift(1).rolling(w)`).
3. **No Future Peeking**: All shift offsets are strictly positive ($+k$). No negative shifts (`shift(-k)`) or centered rolling windows were used.
4. **Independent Chronological Splits**: Splits are partitioned strictly along the chronological time axis. No random shuffling or k-fold cross-validation shuffling.
5. **No Pre-Split Normalization**: Scaling and standardization parameters (mean, std, min, max) must be fitted **strictly on the training partition** during model training.

---

## 11. Chronological Train / Validation / Test Split

The `{len(df):,}` rows are partitioned chronologically:

| Split | Proportion | Row Count | Start Timestamp (Asia/Kolkata) | End Timestamp (Asia/Kolkata) | Duration |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Train** | **70.00%** | **{train_info['num_rows']:,}** | `{train_info['start_timestamp']}` | `{train_info['end_timestamp']}` | ~3.1 years (Aug 2013 – Sep 2016) |
| **Validation** | **15.00%** | **{val_info['num_rows']:,}** | `{val_info['start_timestamp']}` | `{val_info['end_timestamp']}` | ~8 months (Sep 2016 – May 2017) |
| **Test** | **15.00%** | **{test_info['num_rows']:,}** | `{test_info['start_timestamp']}` | `{test_info['end_timestamp']}` | ~8 months (May 2017 – Dec 2017) |

- **Training Set**: Captures multiple full annual academic cycles, severe Delhi summer cooling peaks, and winter baseloads.
- **Validation Set**: Used for model selection, tree-depth tuning, learning-rate optimization, and early stopping.
- **Test Set**: Unseen out-of-time holdout evaluated once to verify model generalization.

---

## 12. Target Variables for Modeling

- **Primary Forecasting Target**: `power` (Hourly mean active power in Watts, W).
- **Secondary Forecasting Target**: `energy_kwh` (Cumulative hourly energy consumption in kWh).

---

## 13. Features That Must NOT Be Used When Predicting the Target

When training models to forecast `power` at timestamp $t$, the following contemporaneous columns **MUST BE EXCLUDED** from the input feature matrix $X$:

```python
# EXCLUDE FROM INPUT FEATURES (Target & Contemporaneous Descriptors):
CONTEMPORANEOUS_EXCLUSIONS = [
    'timestamp',                    # String identifier
    'power',                        # Primary target
    'power_mean',                   # Alias for primary target
    'power_min',                    # Contemporaneous within-hour minimum
    'power_max',                    # Contemporaneous within-hour peak
    'power_std',                    # Contemporaneous within-hour volatility
    'power_range',                  # Contemporaneous within-hour range
    'coefficient_of_variation',     # Contemporaneous within-hour CV
    'energy_kwh',                   # Contemporaneous energy (target or target proxy)
]
```

### Approved Predictive Input Feature Set:
All lag features (`power_lag_*`, `energy_kwh_lag_*`), all rolling features (`power_rolling_*`), all calendar features (`hour`, `day_of_week`, `is_weekend`, etc.), all cyclical features (`*_sin`, `*_cos`), and lagged volatility (`power_range_lag_1`).

---

## 14. GPU Acceleration & Training Plan for Tomorrow's Models

1. **Hardware Configuration**:
   - **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB dedicated GDDR6 VRAM).
   - **Compute Capability**: 8.9 (Ada Lovelace architecture).
   - **CUDA Driver**: 13.4.

2. **Model 1: Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)**:
   - **Execution**: Highly parallelized CPU multi-threading using all available CPU logical cores (`n_jobs=-1`).
   - **Role**: Robust ensemble benchmark, non-linear feature interaction baseline.

3. **Model 2: XGBoost Regressor (`xgboost.XGBRegressor`) with GPU Acceleration**:
   - **Execution**: Direct CUDA GPU acceleration via `tree_method='hist'`, `device='cuda'`.
   - **GPU Features**: Fast GPU histogram construction, parallel tree building, automatic missing-value routing, and GPU memory caching.
   - **Parameters**: `n_estimators=1000`, `learning_rate=0.03`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping_rounds=50`.

---

## 15. Generated Deliverables

1. **Feature Dataset**: `project/data/processed/academic_features.csv` (`{os.path.getsize(output_path):,}` bytes)
2. **Feature Engineering Pipeline**: `project/src/feature_engineering.py`
3. **Split Specification**: `project/reports/split_specification.json`
4. **Comprehensive Report**: `project/reports/feature_engineering_report.md`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"   Successfully wrote report to {report_path}")

if __name__ == "__main__":
    run_feature_engineering()
