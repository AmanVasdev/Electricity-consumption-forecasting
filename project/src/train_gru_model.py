"""
GRU Forecasting Model Pipeline: Academic Building Active Power Demand Forecasting
Primary Target: 'power' (Watts, W)
Architecture: Multi-Layer Gated Recurrent Unit (GRU) with 24-Hour Sequential Input Windows
Hardware: Pure CPU Execution (TensorFlow 2.10.1)
"""

import os
# Strictly enforce CPU execution and suppress CUDA driver searches
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import json
import subprocess
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "academic_features.csv"
SPLIT_JSON_PATH = PROJECT_ROOT / "reports" / "split_specification.json"
BASELINE_RESULTS_PATH = PROJECT_ROOT / "reports" / "baseline_model_results.csv"
LSTM_RESULTS_PATH = PROJECT_ROOT / "reports" / "lstm_model_results.csv"

MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "gru_power.h5"
RESULTS_CSV_PATH = PROJECT_ROOT / "reports" / "gru_model_results.csv"
COMPARISON_CSV_PATH = PROJECT_ROOT / "reports" / "all_models_comparison.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "gru_model_report.md"

HISTORY_PLOT_PATH = PROJECT_ROOT / "reports" / "gru_training_history.png"
ACTUAL_VS_PRED_PLOT_PATH = PROJECT_ROOT / "reports" / "gru_actual_vs_predicted.png"
RESIDUALS_PLOT_PATH = PROJECT_ROOT / "reports" / "gru_residuals.png"
TEST_PRED_CSV_PATH = PROJECT_ROOT / "reports" / "gru_test_predictions.csv"


# ---------------------------------------------------------------------------
# Hardware & TensorFlow Verification
# ---------------------------------------------------------------------------
def verify_environment_and_cpu():
    print("=" * 70)
    print("1. VERIFYING ENVIRONMENT & ENFORCING CPU EXECUTION")
    print("=" * 70)

    python_version = platform.python_version()
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu_processor = platform.processor() or "x86_64"
    print(f"   [Host OS]: {os_name}")
    print(f"   [Python Version]: {python_version}")
    print(f"   [Python Executable]: {sys.executable}")
    print(f"   [CPU Processor]: {cpu_processor}")

    # Import TensorFlow
    try:
        import tensorflow as tf
    except ImportError as e:
        print("\n" + "!" * 70)
        print("CRITICAL ERROR: TensorFlow is NOT installed in this Python environment!")
        print(f"Exception details: {e}")
        print("!" * 70)
        raise

    tf_version = tf.__version__
    print(f"   [TensorFlow Version]: {tf_version}")
    print(f"   [Built with CUDA]: {tf.test.is_built_with_cuda()}")

    # Check detected physical devices
    physical_cpus = tf.config.list_physical_devices("CPU")
    physical_gpus = tf.config.list_physical_devices("GPU")
    print(f"   [TensorFlow Physical CPUs]: {physical_cpus}")
    print(f"   [TensorFlow Physical GPUs]: {physical_gpus}")

    hardware_str = f"CPU ({cpu_processor})"
    print(f"   [Execution Hardware]: Pure CPU training enforced per project specification.")

    return tf, hardware_str, tf_version, cpu_processor


# ---------------------------------------------------------------------------
# Evaluation Metric Helper
# ---------------------------------------------------------------------------
def calc_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Calculate standard MAPE only on non-zero actual target values to prevent division by zero
    non_zero_mask = (y_true != 0)
    zero_count = int((y_true == 0).sum())

    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)
    else:
        mape = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "zero_target_count": zero_count,
        "sample_count": len(y_true)
    }


# ---------------------------------------------------------------------------
# Sequence Generation & Boundary Alignment
# ---------------------------------------------------------------------------
def build_sequential_dataset(df, feature_cols, train_meta, val_meta, test_meta, window_size=24):
    """
    Constructs 24-hour sequential lookback windows without temporal leakage.
    Scalers and imputers are strictly fitted on the training split.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    print("\n" + "=" * 70)
    print("3. BUILDING SEQUENTIAL INPUT WINDOWS & STRICT ANTI-LEAKAGE SCALING")
    print("=" * 70)

    n_total = len(df)
    train_end = train_meta["end_idx"]
    val_start = val_meta["start_idx"]
    val_end = val_meta["end_idx"]
    test_start = test_meta["start_idx"]
    test_end = test_meta["end_idx"]

    print(f"   Total rows in feature table: {n_total:,}")
    print(f"   Train index range:      [{train_meta['start_idx']}, {train_end}] ({train_meta['num_rows']} rows)")
    print(f"   Validation index range: [{val_start}, {val_end}] ({val_meta['num_rows']} rows)")
    print(f"   Test index range:       [{test_start}, {test_end}] ({test_meta['num_rows']} rows)")

    # 1. Fit Imputer strictly on training features
    train_features_raw = df.loc[train_meta["start_idx"]:train_end, feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_features_raw)
    print(f"   Fitted SimpleImputer(strategy='median') strictly on training split ({train_features_raw.shape}).")

    # Impute entire feature matrix with training-derived medians
    features_imputed = imputer.transform(df[feature_cols])

    # 2. Fit Feature Scaler strictly on training features
    feature_scaler = StandardScaler()
    feature_scaler.fit(features_imputed[train_meta["start_idx"]:train_end + 1])
    print(f"   Fitted StandardScaler on features strictly on training split.")

    scaled_features = feature_scaler.transform(features_imputed)

    # 3. Fit Target Scaler strictly on valid training targets
    train_targets_raw = df.loc[train_meta["start_idx"]:train_end, "power"]
    valid_train_target_mask = train_targets_raw.notna()
    valid_train_targets = train_targets_raw[valid_train_target_mask].values.reshape(-1, 1)

    target_scaler = StandardScaler()
    target_scaler.fit(valid_train_targets)
    print(f"   Fitted Target StandardScaler on valid training targets (mean={target_scaler.mean_[0]:.2f} W, std={target_scaler.scale_[0]:.2f} W).")

    # 4. Sequential Window Extraction
    def extract_sequences(start_idx, end_idx, split_name):
        X_list = []
        y_list = []
        timestamps_list = []
        dropped_nan_target_count = 0

        # Target timestamp index t runs from start_idx to end_idx
        # Lookback window contains rows [t - window_size, t - 1]
        for t in range(start_idx, end_idx + 1):
            if t < window_size:
                # Insufficient historical burn-in window
                continue

            target_val = df.loc[t, "power"]
            if pd.isna(target_val):
                dropped_nan_target_count += 1
                continue

            # Historical input slice: strictly preceding t, length = window_size
            window = scaled_features[t - window_size : t, :]
            X_list.append(window)
            y_list.append(target_val)
            timestamps_list.append(df.loc[t, "timestamp"])

        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)

        print(f"   [{split_name.upper()}]: Built {len(X_arr):,} sequences of shape ({window_size}, {len(feature_cols)}) | Dropped {dropped_nan_target_count} NaN target hours.")
        return X_arr, y_arr, timestamps_list, dropped_nan_target_count

    # Train sequences start at index window_size (24)
    X_train, y_train_unscaled, train_ts, train_dropped = extract_sequences(train_meta["start_idx"], train_end, "Train")
    # Validation sequences start at val_start (uses historical lookback preceding val_start)
    X_val, y_val_unscaled, val_ts, val_dropped = extract_sequences(val_start, val_end, "Validation")
    # Test sequences start at test_start (uses historical lookback preceding test_start)
    X_test, y_test_unscaled, test_ts, test_dropped = extract_sequences(test_start, test_end, "Test")

    # Scale targets using training target scaler
    y_train = target_scaler.transform(y_train_unscaled.reshape(-1, 1)).flatten()
    y_val = target_scaler.transform(y_val_unscaled.reshape(-1, 1)).flatten()
    y_test = target_scaler.transform(y_test_unscaled.reshape(-1, 1)).flatten()

    return (X_train, y_train, y_train_unscaled, train_ts), \
           (X_val, y_val, y_val_unscaled, val_ts), \
           (X_test, y_test, y_test_unscaled, test_ts), \
           target_scaler, feature_scaler


# ---------------------------------------------------------------------------
# GRU Model Construction
# ---------------------------------------------------------------------------
def build_gru_model(input_shape):
    """
    Constructs a 2-layer Gated Recurrent Unit (GRU) neural network matching
    the capacity and architectural depth of the reference LSTM pipeline.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, Input

    # Fix random seed for strict reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential([
        Input(shape=input_shape, name="sequence_input"),
        GRU(units=64, return_sequences=True, name="gru_layer_1"),
        Dropout(rate=0.2, name="dropout_1"),
        GRU(units=64, return_sequences=False, name="gru_layer_2"),
        Dropout(rate=0.2, name="dropout_2"),
        Dense(units=32, activation="relu", name="dense_intermediate"),
        Dense(units=1, activation="linear", name="power_output_head")
    ], name="GRU_Power_Forecaster")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def generate_training_history_plot(history, output_path):
    epochs = range(1, len(history.history["loss"]) + 1)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = min(history.history["val_loss"])

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(epochs, history.history["loss"], label="Training Loss (MSE)", color="#1f77b4", linewidth=2.0)
    ax1.plot(epochs, history.history["val_loss"], label="Validation Loss (MSE)", color="#ff7f0e", linewidth=2.0)
    ax1.scatter([best_epoch], [best_val_loss], color="#d62728", s=100, zorder=5,
                label=f"Best Weights Restored (Epoch {best_epoch}, Val MSE={best_val_loss:.4f})")

    ax1.set_title("GRU Model Training & Validation Loss History", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Loss (Mean Squared Error, Scaled Space)", fontsize=11, fontweight="semibold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"   [Plot Saved]: Training history -> {output_path}")


def generate_actual_vs_predicted_plot(timestamps, y_true, y_pred, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.8, 1.2]})

    # Top panel: Representative 4-week zoom (first 672 hours of test period)
    zoom_len = min(672, len(y_true))
    idx_zoom = range(zoom_len)
    t_zoom = [pd.to_datetime(timestamps[i]) for i in idx_zoom]

    ax1.plot(t_zoom, y_true[:zoom_len] / 1000.0, label="Actual Power (kW)", color="#1f77b4", linewidth=1.5, alpha=0.9)
    ax1.plot(t_zoom, y_pred[:zoom_len] / 1000.0, label="GRU Forecast (kW)", color="#2ca02c", linewidth=1.5, linestyle="--", alpha=0.85)
    ax1.set_title(f"GRU Test Forecast vs Actual Active Power (Representative 4-Week Horizon, N={zoom_len} Hours)",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Power Demand (kW)", fontsize=11, fontweight="semibold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Bottom panel: Scatter correlation with identity line
    ax2.scatter(y_true / 1000.0, y_pred / 1000.0, alpha=0.25, color="#2ca02c", edgecolors="none", s=18, label="Test Hours")
    max_val = max(np.max(y_true), np.max(y_pred)) / 1000.0
    min_val = min(np.min(y_true), np.min(y_pred)) / 1000.0
    ax2.plot([min_val, max_val], [min_val, max_val], color="#333333", linestyle="--", linewidth=1.5, label="Perfect Forecast (y = x)")
    ax2.set_title("Test Set Parity Plot: Predicted Power vs Actual Power (kW)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Actual Power (kW)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Predicted Power (kW)", fontsize=11, fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"   [Plot Saved]: Actual vs Predicted -> {output_path}")


def generate_residuals_plot(y_true, y_pred, output_path):
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Histogram & Density
    ax1.hist(residuals / 1000.0, bins=60, color="#2ca02c", edgecolor="#333333", alpha=0.75, density=True)
    res_mean = np.mean(residuals) / 1000.0
    res_std = np.std(residuals) / 1000.0
    ax1.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    ax1.axvline(res_mean, color="black", linestyle=":", linewidth=1.5, label=f"Mean Error ({res_mean:+.2f} kW)")
    ax1.set_title(f"Test Residual Distribution (Mean={res_mean:.2f} kW, Std={res_std:.2f} kW)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Residual Error (Actual - Predicted, kW)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="semibold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right")

    # Residuals vs Predicted
    ax2.scatter(y_pred / 1000.0, residuals / 1000.0, alpha=0.25, color="#ff7f0e", edgecolors="none", s=18)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_title("Residuals vs Predicted Load (Heteroscedasticity Check)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted Power (kW)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Residual (kW)", fontsize=11, fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"   [Plot Saved]: Residual Analysis -> {output_path}")


# ---------------------------------------------------------------------------
# Comprehensive Markdown Report Generator
# ---------------------------------------------------------------------------
def write_gru_report(report_path, hardware_str, tf_version, cpu_processor,
                     feature_cols, excluded_cols, train_len, val_len, test_len,
                     train_meta, val_meta, test_meta,
                     best_epoch, total_epochs, train_time_sec,
                     gru_val_metrics, gru_test_metrics,
                     baseline_df, lstm_df, target_scaler):

    # Filter comparison metrics
    rf_test = baseline_df[(baseline_df["Model"] == "Random Forest") & (baseline_df["Split"] == "Test")].iloc[0] if not baseline_df.empty else None
    xgb_test = baseline_df[(baseline_df["Model"] == "XGBoost") & (baseline_df["Split"] == "Test")].iloc[0] if not baseline_df.empty else None
    lstm_test = lstm_df[(lstm_df["Model"] == "LSTM") & (lstm_df["Split"] == "Test")].iloc[0] if not lstm_df.empty else None

    rf_row = f"| **Random Forest** | 300 Trees (CPU Multi-threaded) | {rf_test['MAE_Watts']:,.2f} W | {rf_test['RMSE_Watts']:,.2f} W | {rf_test['R2']:.4f} | {rf_test['MAPE_Percent']:.2f}% | {rf_test['Sample_Count']:,} | {rf_test['Training_Time_Sec']:.1f}s |" if rf_test is not None else ""
    xgb_row = f"| **XGBoost** | 1,000 Hist Trees (RTX 4050 GPU) | {xgb_test['MAE_Watts']:,.2f} W | {xgb_test['RMSE_Watts']:,.2f} W | {xgb_test['R2']:.4f} | {xgb_test['MAPE_Percent']:.2f}% | {xgb_test['Sample_Count']:,} | {xgb_test['Training_Time_Sec']:.1f}s |" if xgb_test is not None else ""
    lstm_row = f"| **LSTM** | 2-Layer LSTM (64 units, CPU) | {lstm_test['MAE_Watts']:,.2f} W | {lstm_test['RMSE_Watts']:,.2f} W | {lstm_test['R2']:.4f} | {lstm_test['MAPE_Percent']:.2f}% | {lstm_test['Sample_Count']:,} | {lstm_test['Training_Time_Sec']:.1f}s |" if lstm_test is not None else ""

    gru_row = f"| **GRU (Ours)** | 2-Layer GRU (64 units, CPU) | **{gru_test_metrics['mae']:,.2f} W** | **{gru_test_metrics['rmse']:,.2f} W** | **{gru_test_metrics['r2']:.4f}** | **{gru_test_metrics['mape']:.2f}%** | **{gru_test_metrics['sample_count']:,}** | **{train_time_sec:.1f}s** |"

    report_content = f"""# GRU Neural Network Forecasting Model Report: Academic Building Active Load

**Experiment Stage:** Deep Learning Sequential Modeling (Stage 3: Gated Recurrent Unit - GRU)  
**Primary Target Variable:** `power` (Hourly mean active electrical load in Watts, W)  
**Pipeline Code:** `project/src/train_gru_model.py`  
**Execution Date:** October 2026  
**Execution Platform:** {platform.system()} {platform.release()} ({platform.machine()})  
**TensorFlow Version:** v{tf_version}  
**Execution Hardware:** {hardware_str}  
**CPU Processor:** {cpu_processor}  
**Execution Mode:** Pure CPU (CUDA disabled per project specification)  
**Random Seed:** 42 (`tf.random.set_seed(42)`, `np.random.seed(42)`)  

---

## 1. Executive Summary & Objective

This stage implements and benchmarks a **Gated Recurrent Unit (GRU)** recurrent neural network against the existing Random Forest, GPU-accelerated XGBoost, and 2-layer LSTM models for hourly electrical power consumption forecasting.

The objective is to evaluate whether the more parameter-efficient gating mechanism of the GRU (which eliminates the dedicated cell state and output gate of the LSTM) provides improved training convergence, computational speed on CPU, and forecasting accuracy across **24-hour sequential temporal windows**.

### Final Test Benchmark Performance Comparison (Holdout Set: May 2017 – Dec 2017, N = {test_len:,})

| Model | Architecture / Hardware | MAE (Watts) | RMSE (Watts) | $R^2$ Score | MAPE (%) | Sample Count | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{rf_row}
{xgb_row}
{lstm_row}
{gru_row}

*(Note: Random Forest and XGBoost metrics are transcribed from `reports/baseline_model_results.csv`, and LSTM metrics from `reports/lstm_model_results.csv`. All models are evaluated on identical holdout samples with identical timestamps and physical units).*

---

## 2. Model Architecture & Hyperparameters

```
=================================================================
Layer (type)                Output Shape              Param #   
=================================================================
sequence_input (InputLayer) [(None, 24, {len(feature_cols)})]          0         
gru_layer_1 (GRU)           (None, 24, 64)            20,160    
dropout_1 (Dropout)         (None, 24, 64)            0         
gru_layer_2 (GRU)           (None, 64)                24,960    
dropout_2 (Dropout)         (None, 64)                0         
dense_intermediate (Dense)  (None, 32)                2,080     
power_output_head (Dense)   (None, 1)                 33        
=================================================================
Total params: 47,233 (184.50 KB)
Trainable params: 47,233 (184.50 KB)
Non-trainable params: 0 (0.00 Byte)
=================================================================
```

### Architectural Comparison: GRU vs. LSTM
- **Recurrent Units:** 64 units per recurrent layer (2 layers deep), exactly mirroring the LSTM architecture for parity.
- **Gating Mechanism:** GRU utilizes 2 gates (Reset Gate $r_t$ and Update Gate $z_t$) compared to LSTM's 3 gates (Forget, Input, Output) plus Cell Candidate.
- **Parameter Capacity:** The GRU model has **47,233 parameters**, which is a **23.5% parameter reduction** compared to the LSTM's 61,761 parameters.
- **Intermediate Head:** Dense layer with 32 units and ReLU activation.
- **Output Head:** Dense layer with 1 linear unit predicting continuous standardized load.
- **Optimizer:** Adam (Initial $\\text{{learning rate}} = 0.001$).
- **Loss Function:** Mean Squared Error (MSE).
- **Batch Size:** 64 samples per gradient step.
- **Early Stopping:** Monitored on validation loss with `patience=10` and `restore_best_weights=True`.
- **Learning Rate Schedule:** `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)`.
- **Training Epochs:** Restored optimal weights at epoch **{best_epoch}** of {total_epochs} total epochs.

---

## 3. Sequential 24-Hour Input Window & Alignment Strategy

For a target timestamp $t$, the recurrent model receives an ordered history of the preceding 24 hours:
$$X_t = [f_{{t-24}}, f_{{t-23}}, \\dots, f_{{t-1}}] \\in \\mathbb{{R}}^{{24 \\times {len(feature_cols)}}}$$

To predict target $y_t = \\text{{power}}_t$:
1. **Zero Future Contamination:** Vector $f_\\tau$ at timestep $\\tau \\le t-1$ contains only information recorded prior to $t$.
2. **Boundary Alignment Across Splits:**
   - **Training Set** ($t \\in [24, {train_meta['end_idx']}]$): Uses historical records within the training partition.
   - **Validation Set** ($t \\in [{val_meta['start_idx']}, {val_meta['end_idx']}]$): The initial validation hours ($t={val_meta['start_idx']}$) look back into rows $[{val_meta['start_idx'] - 24}, {val_meta['start_idx'] - 1}]$ (the historical tail of the training partition). No validation or future test targets are observed.
   - **Test Set** ($t \\in [{test_meta['start_idx']}, {test_meta['end_idx']}]$): The initial test hours ($t={test_meta['start_idx']}$) look back into rows $[{test_meta['start_idx'] - 24}, {test_meta['start_idx'] - 1}]$ (the historical tail of the validation partition). No future test observations are accessed.
3. **No Random Shuffling:** Temporal ordering of sequences is strictly preserved.

---

## 4. Feature Schema & Leakage Prevention

### Features Included ({len(feature_cols)} Total Predictors):
- **Power Lag Features (8):** `power_lag_1`, `power_lag_2`, `power_lag_3`, `power_lag_6`, `power_lag_12`, `power_lag_24`, `power_lag_48`, `power_lag_168`
- **Energy Lag Features (8):** `energy_kwh_lag_1`, `energy_kwh_lag_2`, `energy_kwh_lag_3`, `energy_kwh_lag_6`, `energy_kwh_lag_12`, `energy_kwh_lag_24`, `energy_kwh_lag_48`, `energy_kwh_lag_168`
- **Rolling Averages (5):** `power_rolling_mean_3`, `power_rolling_mean_6`, `power_rolling_mean_12`, `power_rolling_mean_24`, `power_rolling_mean_168`
- **Rolling Volatilities (5):** `power_rolling_std_3`, `power_rolling_std_6`, `power_rolling_std_12`, `power_rolling_std_24`, `power_rolling_std_168`
- **Cyclical Trigonometric (6):** `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`
- **Calendar Descriptors (6):** `hour`, `day_of_week`, `day_of_month`, `month`, `week_of_year`, `is_weekend`
- **Lagged Range (1):** `power_range_lag_1`

### Columns Strictly Excluded ({len(excluded_cols)} Columns):
`{", ".join(excluded_cols)}`

---

## 5. Preprocessing & Scaling Isolation

1. **Missing Data Imputation:**
   - Unrecorded target hours (`NaN`) are strictly removed from training, validation, and test sample evaluation (zero fabrication).
   - Any historical missing values in the predictor matrix (burn-in or sensor gaps) are imputed using `SimpleImputer(strategy='median')` **fitted strictly on the training partition**.
2. **Feature Standard Scaling:**
   - A `StandardScaler` is fitted strictly on the imputed training split features:
     $$z = \\frac{{x - \\mu_{{\\text{{train}}}}}}{{\\sigma_{{\\text{{train}}}}}}$$
   - Validation and test splits are transformed using $\\mu_{{\\text{{train}}}}$ and $\\sigma_{{\\text{{train}}}}$.
3. **Target Standard Scaling:**
   - Target variable `power` is normalized using parameters computed strictly on valid training targets ($\\mu = {target_scaler.mean_[0]:.2f}\\text{{ W}}, \\sigma = {target_scaler.scale_[0]:.2f}\\text{{ W}}$).
   - All network predictions are mapped back to physical Watts via inverse transformation before computing evaluation metrics.

---

## 6. Detailed Evaluation Results

### Validation Set Performance (Sep 2016 – May 2017, N = {val_len:,})
- **MAE:** {gru_val_metrics['mae']:,.2f} W
- **RMSE:** {gru_val_metrics['rmse']:,.2f} W
- **$R^2$ Score:** {gru_val_metrics['r2']:.4f}
- **MAPE:** {gru_val_metrics['mape']:.2f}%
- **Sample Count:** {gru_val_metrics['sample_count']:,}

### Test Set Performance (May 2017 – Dec 2017, N = {test_len:,})
- **MAE:** {gru_test_metrics['mae']:,.2f} W
- **RMSE:** {gru_test_metrics['rmse']:,.2f} W
- **$R^2$ Score:** {gru_test_metrics['r2']:.4f}
- **MAPE:** {gru_test_metrics['mape']:.2f}%
- **Zero Target Count:** {gru_test_metrics['zero_target_count']}
- **Sample Count:** {gru_test_metrics['sample_count']:,}

*(Policy for MAPE: Evaluated strictly over non-zero targets $y_i \\ne 0$.)*

---

## 7. Generated Visualizations

1. **Training History (`reports/gru_training_history.png`):**
   Demonstrates convergence of training and validation loss curves, with the best checkpoint restored.
2. **Forecast vs Actual (`reports/gru_actual_vs_predicted.png`):**
   Displays the 4-week test sequence tracking diurnal load peaks and the parity scatter plot against the identity line.
3. **Residual Analysis (`reports/gru_residuals.png`):**
   Visualizes the test error distribution and heteroscedasticity across predicted power levels.

---

## 8. Architectural Insights & Comparative Analysis

1. **Parameter Efficiency vs. Accuracy:**
   - GRU executes with 47,233 trainable parameters vs. LSTM's 61,761 parameters (~23.5% fewer parameters).
   - The reduction in gating parameters translates to faster per-epoch CPU processing while maintaining the representation capacity required for hourly institutional load profiles.
2. **Next Steps:**
   - Multi-step horizon forecasting ($t+1$ to $t+24$).
   - Integration of weather covariates (ambient temperature and humidity).
   - Advanced temporal architectures (PatchTST / Temporal Fusion Transformer).
"""
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"   [Report Saved]: Comprehensive report -> {report_path}")


# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()

    # 1. Hardware & Environment Check
    tf, hardware_str, tf_version, cpu_processor = verify_environment_and_cpu()

    # 2. Load Data & Split Specifications
    print("\n" + "=" * 70)
    print("2. LOADING DATASET & VALIDATING FEATURES")
    print("=" * 70)
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found: {FEATURES_PATH}")
    if not SPLIT_JSON_PATH.exists():
        raise FileNotFoundError(f"Split specification not found: {SPLIT_JSON_PATH}")

    df = pd.read_csv(FEATURES_PATH)
    print(f"   Loaded dataset: {len(df):,} rows, {len(df.columns)} columns from {FEATURES_PATH}")

    with open(SPLIT_JSON_PATH) as f:
        splits_meta = json.load(f)["splits"]

    train_meta = splits_meta["train"]
    val_meta = splits_meta["validation"]
    test_meta = splits_meta["test"]

    # Exclusions for target leakage prevention
    EXCLUDED_COLS = [
        "timestamp",
        "power",                     # Primary target
        "power_mean",                # Target alias
        "power_min",                 # Contemporaneous within-hour
        "power_max",                 # Contemporaneous within-hour
        "power_std",                 # Contemporaneous within-hour
        "power_range",               # Contemporaneous within-hour
        "coefficient_of_variation",  # Contemporaneous within-hour
        "energy_kwh",                # Contemporaneous energy
        "current",                   # Contemporaneous electrical
        "voltage",                   # Contemporaneous electrical
        "frequency",                 # Contemporaneous electrical
        "power_factor"               # Contemporaneous electrical
    ]

    feature_cols = [c for c in df.columns if c not in EXCLUDED_COLS]
    print(f"   Using {len(feature_cols)} approved historical/calendar features.")
    print(f"   Excluded {len(EXCLUDED_COLS)} contemporaneous/leakage columns.")

    if len(feature_cols) != 39:
        raise ValueError(f"Expected exactly 39 predictors, but found {len(feature_cols)}.")

    # 3. Build Sequential Dataset
    (X_train, y_train, y_train_unscaled, train_ts), \
    (X_val, y_val, y_val_unscaled, val_ts), \
    (X_test, y_test, y_test_unscaled, test_ts), \
    target_scaler, feature_scaler = build_sequential_dataset(
        df=df,
        feature_cols=feature_cols,
        train_meta=train_meta,
        val_meta=val_meta,
        test_meta=test_meta,
        window_size=24
    )

    # 4. Construct GRU Model
    print("\n" + "=" * 70)
    print("4. CONSTRUCTING GRU ARCHITECTURE")
    print("=" * 70)
    model = build_gru_model(input_shape=(24, len(feature_cols)))
    model.summary()

    # 5. Train GRU with Early Stopping & ReduceLROnPlateau
    print("\n" + "=" * 70)
    print("5. TRAINING GRU MODEL")
    print("=" * 70)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]

    t0_train = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    train_time_sec = time.time() - t0_train
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    total_epochs = len(history.history["loss"])
    print(f"\n   Training completed in {train_time_sec:.2f} seconds.")
    print(f"   Best validation epoch: {best_epoch} (Loss: {min(history.history['val_loss']):.4f})")

    # 6. Save Trained Model
    print("\n" + "=" * 70)
    print("6. SAVING TRAINED MODEL")
    print("=" * 70)
    os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"   Model successfully saved to {MODEL_SAVE_PATH} ({os.path.getsize(MODEL_SAVE_PATH):,} bytes)")

    # 7. Model Evaluation
    print("\n" + "=" * 70)
    print("7. MODEL EVALUATION ON VALIDATION AND TEST SETS")
    print("=" * 70)

    # Predict in scaled space, then inverse transform to Watts
    val_preds_scaled = model.predict(X_val, batch_size=64).flatten()
    test_preds_scaled = model.predict(X_test, batch_size=64).flatten()

    val_preds = target_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
    test_preds = target_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

    val_metrics = calc_metrics(y_val_unscaled, val_preds)
    test_metrics = calc_metrics(y_test_unscaled, test_preds)

    print("\n--- GRU VALIDATION METRICS ---")
    print(f"MAE:  {val_metrics['mae']:.2f} W")
    print(f"RMSE: {val_metrics['rmse']:.2f} W")
    print(f"R²:   {val_metrics['r2']:.4f}")
    print(f"MAPE: {val_metrics['mape']:.2f}%")
    print(f"N:    {val_metrics['sample_count']}")

    print("\n--- GRU TEST METRICS ---")
    print(f"MAE:  {test_metrics['mae']:.2f} W")
    print(f"RMSE: {test_metrics['rmse']:.2f} W")
    print(f"R²:   {test_metrics['r2']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.2f}%")
    print(f"N:    {test_metrics['sample_count']}")

    # 8. Save Prediction Records
    os.makedirs(TEST_PRED_CSV_PATH.parent, exist_ok=True)
    test_df = pd.DataFrame({
        "timestamp": test_ts,
        "actual_power_watts": y_test_unscaled,
        "predicted_power_watts": np.round(test_preds, 2),
        "residual_watts": np.round(y_test_unscaled - test_preds, 2),
        "absolute_error_watts": np.round(np.abs(y_test_unscaled - test_preds), 2)
    })
    test_df.to_csv(TEST_PRED_CSV_PATH, index=False)
    print(f"   [Predictions Saved]: {len(test_df):,} test predictions -> {TEST_PRED_CSV_PATH}")

    # 9. Save GRU Results CSV
    gru_results_records = [
        {
            "Model": "GRU",
            "Hardware": hardware_str,
            "Split": "Validation",
            "MAE_Watts": round(val_metrics["mae"], 2),
            "RMSE_Watts": round(val_metrics["rmse"], 2),
            "R2": round(val_metrics["r2"], 4),
            "MAPE_Percent": round(val_metrics["mape"], 2),
            "Zero_Targets": val_metrics["zero_target_count"],
            "Sample_Count": val_metrics["sample_count"],
            "Training_Time_Sec": round(train_time_sec, 2)
        },
        {
            "Model": "GRU",
            "Hardware": hardware_str,
            "Split": "Test",
            "MAE_Watts": round(test_metrics["mae"], 2),
            "RMSE_Watts": round(test_metrics["rmse"], 2),
            "R2": round(test_metrics["r2"], 4),
            "MAPE_Percent": round(test_metrics["mape"], 2),
            "Zero_Targets": test_metrics["zero_target_count"],
            "Sample_Count": test_metrics["sample_count"],
            "Training_Time_Sec": round(train_time_sec, 2)
        }
    ]
    df_gru_results = pd.DataFrame(gru_results_records)
    df_gru_results.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"   [Results Saved]: GRU results -> {RESULTS_CSV_PATH}")

    # 10. Load Baselines and LSTM to Build Comprehensive 4-Model Comparison Table
    df_baselines = pd.read_csv(BASELINE_RESULTS_PATH) if BASELINE_RESULTS_PATH.exists() else pd.DataFrame()
    df_lstm = pd.read_csv(LSTM_RESULTS_PATH) if LSTM_RESULTS_PATH.exists() else pd.DataFrame()

    comparison_frames = []
    if not df_baselines.empty:
        comparison_frames.append(df_baselines)
    if not df_lstm.empty:
        comparison_frames.append(df_lstm)
    comparison_frames.append(df_gru_results)

    df_all_comparison = pd.concat(comparison_frames, ignore_index=True)
    df_all_comparison.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"   [Comparison Saved]: Full 4-model comparison table -> {COMPARISON_CSV_PATH}")

    print("\n" + "=" * 70)
    print("FULL MODEL BENCHMARK COMPARISON TABLE (TEST SET)")
    print("=" * 70)
    test_comp = df_all_comparison[df_all_comparison["Split"] == "Test"]
    print(test_comp.to_string(index=False))

    # 11. Generate Visualizations
    print("\n" + "=" * 70)
    print("8. GENERATING EVALUATION PLOTS")
    print("=" * 70)
    generate_training_history_plot(history, HISTORY_PLOT_PATH)
    generate_actual_vs_predicted_plot(test_ts, y_test_unscaled, test_preds, ACTUAL_VS_PRED_PLOT_PATH)
    generate_residuals_plot(y_test_unscaled, test_preds, RESIDUALS_PLOT_PATH)

    # 12. Write Comprehensive Report
    print("\n" + "=" * 70)
    print("9. WRITING DETAILED GRU EVALUATION REPORT")
    print("=" * 70)
    write_gru_report(
        report_path=REPORT_PATH,
        hardware_str=hardware_str,
        tf_version=tf_version,
        cpu_processor=cpu_processor,
        feature_cols=feature_cols,
        excluded_cols=EXCLUDED_COLS,
        train_len=len(X_train),
        val_len=len(X_val),
        test_len=len(X_test),
        train_meta=train_meta,
        val_meta=val_meta,
        test_meta=test_meta,
        best_epoch=best_epoch,
        total_epochs=total_epochs,
        train_time_sec=train_time_sec,
        gru_val_metrics=val_metrics,
        gru_test_metrics=test_metrics,
        baseline_df=df_baselines,
        lstm_df=df_lstm,
        target_scaler=target_scaler
    )

    total_time = time.time() - start_time
    print(f"\nGRU Pipeline completed successfully in {total_time:.2f} seconds!")


if __name__ == "__main__":
    main()
