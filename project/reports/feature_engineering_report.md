# Feature Engineering Report: Electricity Consumption & Load Forecasting

**Project Stage:** Feature Engineering  
**Dataset:** Academic Building Mains (`acad_build_mains.csv`)  
**Execution Date:** September 29, 2026  
**Script Executed:** `project/src/feature_engineering.py`  
**Execution Environment:** `.\.venv\Scripts\python.exe`  
**Hardware Profile:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM, CUDA 13.4)  

---

## 1. Input Dataset
- **File Path:** `project\data\processed\academic_hourly.csv`
- **Dimensions:** `38,520` rows $\times$ `17` columns
- **Temporal Granularity:** Regular hourly resolution (`freq='1h'`)
- **Date Range:** `2013-08-10 00:00:00+05:30` to `2017-12-31 23:00:00+05:30` (1,605 days)

---

## 2. Output Dataset
- **File Path:** `project\data\processed\academic_features.csv`
- **Dimensions:** `38,520` rows $\times$ `52` columns
- **File Size:** `30,855,702` bytes (~`29.43` MB)
- **Integrity:** Strict chronological ordering; zero duplicate timestamps; no overwriting of previous artifacts.

---

## 3. Column Count Comparison
- **Original Number of Columns:** `17`
- **New Engineered Features Added:** `35`
- **Final Number of Columns:** `52`

---

## 4. List of All Engineered Features

The `35` engineered features are grouped into five functional categories:

| Category | Count | Feature Names | Mathematical / Logical Definition |
| :--- | :---: | :--- | :--- |
| **Power Lags** | 8 | `power_lag_1`, `power_lag_2`, `power_lag_3`, `power_lag_6`, `power_lag_12`, `power_lag_24`, `power_lag_48`, `power_lag_168` | $\text{power}_{t-k} = \text{power.shift}(k)$ |
| **Energy Lags** | 8 | `energy_kwh_lag_1`, `energy_kwh_lag_2`, `energy_kwh_lag_3`, `energy_kwh_lag_6`, `energy_kwh_lag_12`, `energy_kwh_lag_24`, `energy_kwh_lag_48`, `energy_kwh_lag_168` | $\text{energy\_kwh}_{t-k} = \text{energy\_kwh.shift}(k)$ |
| **Rolling Means** | 5 | `power_rolling_mean_3`, `power_rolling_mean_6`, `power_rolling_mean_12`, `power_rolling_mean_24`, `power_rolling_mean_168` | $\frac{1}{W}\sum_{i=1}^{W} \text{power}_{t-i} = \text{power.shift}(1).\text{rolling}(W).\text{mean}()$ |
| **Rolling Std** | 5 | `power_rolling_std_3`, `power_rolling_std_6`, `power_rolling_std_12`, `power_rolling_std_24`, `power_rolling_std_168` | $\sqrt{\frac{1}{W-1}\sum_{i=1}^{W} (\text{power}_{t-i} - \bar{P})^2} = \text{power.shift}(1).\text{rolling}(W).\text{std}()$ |
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
   - Exactly 168 hours ($7 \times 24\text{h}$) prior. Represents the identical hour on the same day of the preceding week, capturing institutional schedule alignment (e.g. comparing Monday 10:00 AM to the prior Monday 10:00 AM).
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
$$\text{hour\_sin} = \sin\left(\frac{2\pi \times \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \times \text{hour}}{24}\right)$$
$$\text{day\_of\_week\_sin} = \sin\left(\frac{2\pi \times \text{day\_of\_week}}{7}\right), \quad \text{day\_of\_week\_cos} = \cos\left(\frac{2\pi \times \text{day\_of\_week}}{7}\right)$$
$$\text{month\_sin} = \sin\left(\frac{2\pi \times (\text{month} - 1)}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \times (\text{month} - 1)}{12}\right)$$

---

## 8. Power-Derived Features

1. **`power_range`**:
   $$\text{power\_range} = \text{power\_max} - \text{power\_min}$$
   Measures the span of instantaneous power variation within that hour.
2. **`coefficient_of_variation` (CV)**:
   $$\text{CV} = \frac{\text{power\_std}}{\text{power\_mean}}$$
   Normalized dimensionless measure of relative within-hour load dispersion. Handled with zero-division safeguard: if $\text{power\_mean} = 0$, $\text{CV} = 0.0$.
3. **`power_range_lag_1`**:
   $$\text{power\_range\_lag\_1} = \text{power\_range.shift}(1)$$
   The previous hour's within-hour volatility, providing a strictly historical measure of load fluctuation for forecasting the current hour.

---

## 9. Missing-Value Handling Strategy

1. **Preservation of Genuine Zeros**:
   - 107 full hours with $\text{power} = 0.0$ are preserved as real physical outage/shutdown events. They are **not** treated as missing.
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

The `38,520` rows are partitioned chronologically:

| Split | Proportion | Row Count | Start Timestamp (Asia/Kolkata) | End Timestamp (Asia/Kolkata) | Duration |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Train** | **70.00%** | **26,964** | `2013-08-10 00:00:00+05:30` | `2016-09-06 11:00:00+05:30` | ~3.1 years (Aug 2013 – Sep 2016) |
| **Validation** | **15.00%** | **5,778** | `2016-09-06 12:00:00+05:30` | `2017-05-05 05:00:00+05:30` | ~8 months (Sep 2016 – May 2017) |
| **Test** | **15.00%** | **5,778** | `2017-05-05 06:00:00+05:30` | `2017-12-31 23:00:00+05:30` | ~8 months (May 2017 – Dec 2017) |

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

1. **Feature Dataset**: `project/data/processed/academic_features.csv` (`30,855,702` bytes)
2. **Feature Engineering Pipeline**: `project/src/feature_engineering.py`
3. **Split Specification**: `project/reports/split_specification.json`
4. **Comprehensive Report**: `project/reports/feature_engineering_report.md`
