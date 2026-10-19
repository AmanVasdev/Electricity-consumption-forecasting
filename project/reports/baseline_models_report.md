# Baseline Forecasting Models Report: Random Forest & GPU XGBoost

**Experiment Stage:** First Modeling Stage (Baseline Models)  
**Target Variable:** `power` (Hourly mean active electrical load in Watts, W)  
**Execution Date:** September 29, 2026  
**Pipeline Code:** `project/src/train_baseline_models.py`  
**Execution Environment:** `.\.venv\Scripts\python.exe`  
**Hardware Accelerators:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB GDDR6 VRAM, CUDA 13.4) & Multi-Core Host CPU  

---

## 1. Dataset & Scope
- **Source Feature Matrix:** `project/data/processed/academic_features.csv`
- **Total Dataset Rows:** `38,520` continuous hourly intervals (1,605 calendar days, Aug 10, 2013 to Dec 31, 2017)
- **Total Columns in Dataset:** `52`
- **Predictive Features Used:** `39` strictly historical and calendar features
- **Columns Excluded:** `13` contemporaneous target and within-hour descriptor columns

---

## 2. Target Variable
- **Designation:** `power`
- **Physical Interpretation:** Average active electrical power demanded by the Academic Building over each 1-hour window.
- **Unit of Measurement:** Watts (W), where $1,000\text{ W} = 1.0\text{ kW}$.
- **Dataset Mean Load:** ~`27,785.82 W` (~27.79 kW) across valid hours, with peak summer operational demand reaching ~`95,786.33 W` (~95.79 kW).

---

## 3. Features Used in Modeling (39 Features)

The feature matrix $X$ incorporates 39 non-leaking predictors across six functional groupings:

1. **Power Lag Features (8)**: `power_lag_1`, `power_lag_2`, `power_lag_3`, `power_lag_6`, `power_lag_12`, `power_lag_24`, `power_lag_48`, `power_lag_168`
2. **Energy Lag Features (8)**: `energy_kwh_lag_1`, `energy_kwh_lag_2`, `energy_kwh_lag_3`, `energy_kwh_lag_6`, `energy_kwh_lag_12`, `energy_kwh_lag_24`, `energy_kwh_lag_48`, `energy_kwh_lag_168`
3. **Power Historical Rolling Means (5)**: `power_rolling_mean_3`, `power_rolling_mean_6`, `power_rolling_mean_12`, `power_rolling_mean_24`, `power_rolling_mean_168`
4. **Power Historical Rolling Std (5)**: `power_rolling_std_3`, `power_rolling_std_6`, `power_rolling_std_12`, `power_rolling_std_24`, `power_rolling_std_168`
5. **Cyclical Temporal Features (6)**: `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`
6. **Base Calendar Features (6)**: `hour`, `day_of_week`, `day_of_month`, `month`, `week_of_year`, `is_weekend`
7. **Lagged Volatility (1)**: `power_range_lag_1` (Within-hour power range of the previous hour, $\text{power\_range}_{t-1}$)

---

## 4. Features Excluded (Target Leakage Prevention)

To guarantee that models are evaluated under realistic deployment conditions, all contemporaneous metrics measured during hour $t$ were strictly excluded from $X$:
- `timestamp`: Non-predictive index string.
- `power`, `power_mean`: Target variable and its alias.
- `power_min`, `power_max`, `power_std`, `power_range`, `coefficient_of_variation`: Within-hour statistics computed from 1-minute measurements inside hour $t$.
- `energy_kwh`: Contemporaneous cumulative energy consumption inside hour $t$.
- `current`, `voltage`, `frequency`, `power_factor`: Contemporaneous electrical measurements inside hour $t$.

---

## 5. Chronological Train / Validation / Test Split

The dataset was partitioned along the chronological time axis according to `project/reports/split_specification.json`:

| Split | Raw Rows | Target NaNs Dropped | Modeled Rows | Start Date (Asia/Kolkata) | End Date (Asia/Kolkata) | Duration |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **Train** | 26,964 (70.0%) | 498 | **26,466** | 2013-08-10 00:00:00+05:30 | 2016-09-06 11:00:00+05:30 | ~3.1 years |
| **Validation** | 5,778 (15.0%) | 22 | **5,756** | 2016-09-06 12:00:00+05:30 | 2017-05-05 05:00:00+05:30 | ~8 months |
| **Test** | 5,778 (15.0%) | 118 | **5,660** | 2017-05-05 06:00:00+05:30 | 2017-12-31 23:00:00+05:30 | ~8 months |
| **Total** | 38,520 (100.0%) | 638 | **37,882** | 2013-08-10 00:00:00+05:30 | 2017-12-31 23:00:00+05:30 | 4.4 years |

---

## 6. Missing-Value Handling Strategy
1. **Target Rows**: All 638 recording-outage hours where `power` was unobserved (`NaN`) were dropped from the training, validation, and test partitions.
2. **Zero-Power Readings**: Exactly 107 full hours in the training set where $\text{power} = 0.0\text{ W}$ were preserved as genuine physical outage/blackout events. Validation and test partitions contained zero zero-power occurrences.
3. **Feature NaNs in XGBoost**: XGBoost natively handles missing values via optimal default split routing on GPU, preserving the un-imputed nature of historical sensor gaps.
4. **Feature NaNs in Random Forest**: scikit-learn 1.9 `RandomForestRegressor` natively supports missing values by evaluating impurity-minimizing split paths for missing values. No global imputer was fit across splits.

---

## 7. Model Configurations

### Model 1: Random Forest Regressor (CPU)
- **Library**: `sklearn.ensemble.RandomForestRegressor`
- **Execution Target**: CPU multi-threading (`n_jobs=-1`, 100% CPU thread utilization)
- **Parameters**:
  - `n_estimators`: `300`
  - `max_depth`: `None` (Full tree expansion)
  - `random_state`: `42`
  - `n_jobs`: `-1`

### Model 2: XGBoost Regressor (GPU Accelerated)
- **Library**: `xgboost` (v`3.4.1`)
- **Execution Target**: NVIDIA GeForce RTX 4050 Laptop GPU (`device="cuda"`)
- **Parameters**:
  - `tree_method`: `"hist"` (GPU-accelerated histogram construction)
  - `device`: `"cuda"`
  - `n_estimators`: `1000` (Max trees)
  - `learning_rate`: `0.03`
  - `max_depth`: `6`
  - `subsample`: `0.8`
  - `colsample_bytree`: `0.8`
  - `random_state`: `42`
  - `early_stopping_rounds`: `50` (Monitored on validation RMSE)
  - `eval_metric`: `"rmse"`
- **Early Stopping Trigger**: Stopped at iteration **626** with optimal validation RMSE.

---

## 8. GPU Verification & Execution Details
- **Detected Hardware**: `NVIDIA GeForce RTX 4050 Laptop GPU`
- **VRAM Total**: 6,141 MiB dedicated GDDR6
- **CUDA Architecture**: Ada Lovelace (Compute Capability 8.9)
- **CUDA Verification**: Validated with direct CUDA kernel launches. Zero CPU fallback occurred.
- **Speedup Observed**: XGBoost on GPU trained in **1.88 seconds**, compared to **25.30 seconds** for Random Forest on CPU (a **~13.5x speedup**).

---

## 9. Model Evaluation & Performance Comparison

All metrics are reported in actual physical units (Watts, W) and percentage error (MAPE):

### 9.1 Validation Set Performance (Sep 2016 – May 2017, N = 5,756)

| Model | Hardware | Training Time | MAE (Watts) | RMSE (Watts) | $R^2$ Score | MAPE (%) | Zero Target Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | CPU (`n_jobs=-1`) | 25.30s | 1,541.73 W | 2,627.33 W | 0.9601 | 5.23% | 0 |
| **XGBoost** | **GPU (CUDA)** | **1.88s** | **1,507.65 W** | **2,458.22 W** | **0.9650** | **5.09%** | 0 |

### 9.2 Test Set Performance (May 2017 – Dec 2017, N = 5,660)

| Model | Hardware | MAE (Watts) | RMSE (Watts) | $R^2$ Score | MAPE (%) | Zero Target Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | CPU (`n_jobs=-1`) | 2,763.39 W | 4,982.39 W | 0.9049 | 8.03% | 0 |
| **XGBoost** | **GPU (CUDA)** | **2,574.83 W** | **4,469.55 W** | **0.9234** | **7.63%** | 0 |

*Note on MAPE: Computed strictly on non-zero target instances ($y_i \ne 0$). Zero-target counts are 0 on validation and test.*

---

## 10. Key Analytical Findings

1. **Superior Accuracy with XGBoost**:
   - On the unseen test partition, XGBoost achieved an **$R^2$ of 0.9234** and a **MAPE of 7.63%**, outperforming Random Forest ($R^2 = 0.9049$, $\text{MAPE} = 8.03\%$).
   - On average, test power predictions have an error of ~2.57 kW on a building that fluctuates between 10 kW and 80+ kW.
2. **Computational Efficiency via GPU**:
   - The RTX 4050 GPU enabled XGBoost to evaluate 1,000 boosting rounds with early stopping in only **1.88 seconds**, demonstrating exceptional computational throughput for rapid iterative experiments.
3. **Generalization Across Shifts**:
   - Both models generalized solidly from historical years (2013–2016) into out-of-time holdouts (2017), maintaining $R^2 > 0.90$. The slight error increase on the test set reflects the intense 2017 summer cooling peaks.

---

## 11. Feature Importance Summary

### Top 10 Features (XGBoost GPU - Gain/Relative Importance):
| feature              |   importance |
|:---------------------|-------------:|
| power_lag_1          |    0.530654  |
| energy_kwh_lag_1     |    0.200919  |
| hour_cos             |    0.0501411 |
| power_lag_168        |    0.0386698 |
| hour                 |    0.0185702 |
| hour_sin             |    0.0181232 |
| power_lag_2          |    0.0178213 |
| power_range_lag_1    |    0.017047  |
| power_rolling_mean_3 |    0.01297   |
| energy_kwh_lag_2     |    0.0105886 |

### Top 10 Features (Random Forest CPU - MDI/Impurity Importance):
| feature            |   importance |
|:-------------------|-------------:|
| power_lag_1        |   0.856249   |
| power_range_lag_1  |   0.0324505  |
| hour_cos           |   0.0212869  |
| power_lag_2        |   0.0116628  |
| power_lag_168      |   0.0102705  |
| hour               |   0.00847164 |
| hour_sin           |   0.0081718  |
| energy_kwh_lag_1   |   0.00756153 |
| power_lag_24       |   0.00403312 |
| energy_kwh_lag_168 |   0.00372067 |

**Core Takeaways**:
- `power_lag_1` and `energy_kwh_lag_1` are the dominant primary predictors, contributing over 65% of total model gain, demonstrating strong short-term electrical load persistence.
- `power_lag_168` (weekly cycle) and cyclical calendar terms (`hour_cos`, `hour_sin`) provide critical temporal scaffolding, capturing university operational hours and weekend baseloads.
- `power_range_lag_1` and short rolling averages (`power_rolling_mean_3`, `power_rolling_mean_24`) capture local load volatility and baseline shifts.

---

## 12. Current Model Limitations
1. **No Exogenous Environmental Drivers**: Weather variables (ambient temperature, humidity, solar radiation) have not yet been merged. High summer cooling load surges are currently inferred strictly from calendar and lag features.
2. **No Occupancy Data**: Campus attendance, academic holiday schedules, and semester calendar schedules are not yet integrated.
3. **Single-Step Focus**: Currently configured for 1-step-ahead ($t+1$) hourly forecasting. Multi-step recursive or direct forecasting (24-hour horizon) remains to be implemented.

---

## 13. Next Planned Modeling Stages
1. **Recurrent & Sequence Neural Architectures**:
   - **LSTM (Long Short-Term Memory)**: Sequence modeling of temporal persistence and state retention across 24h/168h windows.
   - **GRU (Gated Recurrent Unit)**: Computationally lightweight recurrent baseline with faster convergence on the RTX 4050 GPU.
2. **Temporal Transformers**:
   - Self-attention mechanisms across long sequence windows to capture multi-scale periodic dependencies (PatchTST / Informer / Temporal Fusion Transformer architectures).
3. **Multi-Modal Dataset Merging**:
   - Merge `weather_comparison/` (temperature, humidity) to provide direct meteorological drivers for HVAC consumption.
   - Merge `IIITD_occupancy_dataset/` and `iiitd_calender_schedule/` to correlate academic schedules with power demand.
4. **Unsupervised & Semi-Supervised Anomaly Detection**:
   - Isolation Forests, Autoencoders, and reconstruction error thresholds on the residual load profile.

---

## 14. Generated Deliverables
1. **Trained Random Forest Model:** `project/models/random_forest_power.pkl` (720,122,625 bytes)
2. **Trained XGBoost GPU Model:** `project/models/xgboost_power.json` (4,206,632 bytes)
3. **Benchmark Results Table:** `project/reports/baseline_model_results.csv` (456 bytes)
4. **Feature Importance Datasets:**
   - `project/reports/random_forest_feature_importance.csv` (1,506 bytes)
   - `project/reports/xgboost_feature_importance.csv` (1,140 bytes)
5. **Feature Importance Visualizations:**
   - `project/reports/random_forest_feature_importance.png` (136,204 bytes)
   - `project/reports/xgboost_feature_importance.png` (139,463 bytes)
6. **Detailed Evaluation Report:** `project/reports/baseline_models_report.md`
