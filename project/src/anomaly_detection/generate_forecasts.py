"""
Electricity Consumption Forecasting & Anomaly Detection Pipeline
Component: Forecast and Residual Generation for Anomaly Detection
File: project/src/anomaly_detection/generate_forecasts.py

Responsibilities:
1. Load existing feature dataset (data/processed/academic_features.csv).
2. Load saved baseline XGBoost forecasting model (models/xgboost_power.json).
3. Validate all 39 approved predictive features and their exact sequence.
4. Reproduce the approved chronological train/validation/test split.
5. Generate validation and test set predictions using the best iteration.
6. Compute residuals (actual_power - predicted_power) on available target rows.
7. Preserve unrecorded sensor gap rows (missing actual targets) with NaN residuals.
8. Save validation_predictions.csv and test_predictions.csv to data/anomaly_detection/.
9. Output comprehensive verification metrics and assertions.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb

# ---------------------------------------------------------------------------
# Path Configuration (Dynamically resolved relative to project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # project root: project/

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "academic_features.csv"
SPLIT_JSON_PATH = PROJECT_ROOT / "reports" / "split_specification.json"
XGB_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_power.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "anomaly_detection"
VAL_PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

# ---------------------------------------------------------------------------
# Exact 39 Approved Predictive Features (Order Preserved)
# ---------------------------------------------------------------------------
FEATURES: List[str] = [
    "hour", "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
    "power_lag_1", "power_lag_2", "power_lag_3", "power_lag_6", "power_lag_12",
    "power_lag_24", "power_lag_48", "power_lag_168",
    "energy_kwh_lag_1", "energy_kwh_lag_2", "energy_kwh_lag_3",
    "energy_kwh_lag_6", "energy_kwh_lag_12", "energy_kwh_lag_24",
    "energy_kwh_lag_48", "energy_kwh_lag_168",
    "power_rolling_mean_3", "power_rolling_std_3",
    "power_rolling_mean_6", "power_rolling_std_6",
    "power_rolling_mean_12", "power_rolling_std_12",
    "power_rolling_mean_24", "power_rolling_std_24",
    "power_rolling_mean_168", "power_rolling_std_168",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos", "power_range_lag_1"
]

# Excluded columns that must not be used as model inputs
EXCLUDED_COLS: List[str] = [
    "timestamp",
    "power",
    "power_mean",
    "power_min",
    "power_max",
    "power_std",
    "power_range",
    "coefficient_of_variation",
    "energy_kwh",
    "current",
    "voltage",
    "frequency",
    "power_factor"
]

# Chronological split definitions (standard project boundaries)
EXPECTED_SPLITS = {
    "train": {"start_idx": 0, "end_idx": 26963, "num_rows": 26964, "nan_power": 498},
    "validation": {"start_idx": 26964, "end_idx": 32741, "num_rows": 5778, "nan_power": 22},
    "test": {"start_idx": 32742, "end_idx": 38519, "num_rows": 5778, "nan_power": 118}
}


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the processed feature dataset with verification."""
    if not file_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at: {file_path}")
    
    print(f"Loading feature dataset: {file_path}")
    df = pd.read_csv(file_path)
    print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    if len(df) != 38520:
        raise ValueError(f"Expected 38,520 rows in {file_path.name}, but found {len(df):,}")
    if "power" not in df.columns:
        raise KeyError(f"Target column 'power' missing from dataset.")
    if "timestamp" not in df.columns:
        raise KeyError(f"Timestamp column 'timestamp' missing from dataset.")
        
    return df


def load_split_specification(split_path: Path) -> Dict:
    """Load split specification from JSON or use exact default boundaries."""
    if split_path.exists():
        print(f"Loading split specification from: {split_path}")
        with open(split_path, "r", encoding="utf-8") as f:
            meta = json.load(f)["splits"]
        return meta
    else:
        print(f"Warning: {split_path} not found. Falling back to exact canonical boundaries.")
        return EXPECTED_SPLITS


def load_and_verify_model(model_path: Path) -> Tuple[xgb.XGBRegressor, int]:
    """Load the saved XGBoost model and verify feature names, count, and best iteration."""
    if not model_path.exists():
        raise FileNotFoundError(f"Saved XGBoost model not found at: {model_path}")
    
    print(f"\nLoading saved XGBoost model: {model_path}")
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    
    booster = model.get_booster()
    num_features = booster.num_features()
    feature_names = booster.feature_names
    
    print(f"   Booster feature count: {num_features}")
    if num_features != len(FEATURES):
        raise ValueError(
            f"Model feature count mismatch: booster has {num_features}, expected {len(FEATURES)}"
        )
    
    if feature_names != FEATURES:
        # Check mismatches
        missing = set(FEATURES) - set(feature_names or [])
        extra = set(feature_names or []) - set(FEATURES)
        raise ValueError(
            f"Feature mismatch between model and approved feature list.\n"
            f"Missing: {missing}\nExtra: {extra}"
        )
    print("   [Feature Verification]: SUCCESS! All 39 features match model booster exactly.")
    
    # Verify best_iteration semantics
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        raw_attr = booster.attr("best_iteration")
        best_iter = int(raw_attr) if raw_attr is not None else None
    
    print(f"   Best iteration: {best_iter}")
    return model, best_iter


def generate_forecasts_and_residuals(
    df_split: pd.DataFrame,
    split_name: str,
    model: xgb.XGBRegressor,
    best_iter: int
) -> pd.DataFrame:
    """
    Generate predictions and compute residuals while preserving rows with missing targets.
    
    Formula:
        residual_watts = actual_power_watts - predicted_power_watts
        absolute_error_watts = abs(residual_watts)
    """
    print(f"\nGenerating forecasts for '{split_name}' split ({len(df_split):,} rows)...")
    
    # Feature matrix (XGBoost natively handles missing feature values if any)
    X = df_split[FEATURES]
    
    # Generate predictions using best iteration range
    if best_iter is not None:
        preds = model.predict(X, iteration_range=(0, best_iter + 1))
    else:
        preds = model.predict(X)
        
    preds = np.asarray(preds, dtype=np.float64)
    
    # Verify prediction sanity
    if not np.all(np.isfinite(preds)):
        nan_preds = np.isnan(preds).sum()
        inf_preds = np.isinf(preds).sum()
        raise ValueError(
            f"Model produced invalid predictions: {nan_preds} NaNs, {inf_preds} Infs"
        )
    
    actual_power = df_split["power"].values.astype(np.float64)
    nan_target_mask = np.isnan(actual_power)
    nan_target_count = int(nan_target_mask.sum())
    valid_target_count = len(actual_power) - nan_target_count
    
    # Compute residuals: positive = consumption higher than expected, negative = lower
    residuals = np.where(nan_target_mask, np.nan, actual_power - preds)
    abs_errors = np.where(nan_target_mask, np.nan, np.abs(residuals))
    
    # Construct structured DataFrame
    predictions_df = pd.DataFrame({
        "timestamp": df_split["timestamp"].values,
        "actual_power_watts": np.round(actual_power, 2),
        "predicted_power_watts": np.round(preds, 2),
        "residual_watts": np.round(residuals, 2),
        "absolute_error_watts": np.round(abs_errors, 2)
    })
    
    # Evaluation metrics on clean (valid target) subset
    if valid_target_count > 0:
        valid_y = actual_power[~nan_target_mask]
        valid_preds = preds[~nan_target_mask]
        mae = mean_absolute_error(valid_y, valid_preds)
        rmse = root_mean_squared_error(valid_y, valid_preds)
        r2 = r2_score(valid_y, valid_preds)
        print(f"   [Clean Evaluation ({valid_target_count:,} rows)]:")
        print(f"      MAE:  {mae:10.2f} W")
        print(f"      RMSE: {rmse:10.2f} W")
        print(f"      R²:   {r2:10.4f}")
    
    print(f"   [Missing Targets Preserved]: {nan_target_count} rows marked with NaN residual")
    print(f"   [Total Output Rows]:         {len(predictions_df):,}")
    
    return predictions_df


def verify_output_file(
    csv_path: Path,
    expected_rows: int,
    expected_nan_targets: int,
    expected_start_ts: str,
    expected_end_ts: str
) -> None:
    """Read back and strictly verify the generated CSV against requirements."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected output file not found: {csv_path}")
    
    df_read = pd.read_csv(csv_path)
    
    # Verify row count
    if len(df_read) != expected_rows:
        raise ValueError(
            f"Row count mismatch in {csv_path.name}: got {len(df_read):,}, expected {expected_rows:,}"
        )
        
    # Verify columns
    expected_cols = [
        "timestamp", "actual_power_watts", "predicted_power_watts",
        "residual_watts", "absolute_error_watts"
    ]
    if list(df_read.columns) != expected_cols:
        raise ValueError(
            f"Column schema mismatch in {csv_path.name}: {list(df_read.columns)} vs {expected_cols}"
        )
        
    # Verify missing target counts
    actual_nan_targets = int(df_read["actual_power_watts"].isna().sum())
    if actual_nan_targets != expected_nan_targets:
        raise ValueError(
            f"Missing target count mismatch in {csv_path.name}: got {actual_nan_targets}, expected {expected_nan_targets}"
        )
        
    # Verify residual NaNs match target NaNs
    actual_nan_res = int(df_read["residual_watts"].isna().sum())
    if actual_nan_res != expected_nan_targets:
        raise ValueError(
            f"Residual NaN count mismatch in {csv_path.name}: got {actual_nan_res}, expected {expected_nan_targets}"
        )
        
    # Verify predictions are 100% complete and finite
    nan_preds = int(df_read["predicted_power_watts"].isna().sum())
    if nan_preds != 0:
        raise ValueError(f"Found {nan_preds} NaN values in predicted_power_watts of {csv_path.name}")
        
    # Verify start and end timestamps
    start_ts = str(df_read["timestamp"].iloc[0])
    end_ts = str(df_read["timestamp"].iloc[-1])
    if start_ts != expected_start_ts:
        raise ValueError(f"Start timestamp mismatch: got {start_ts}, expected {expected_start_ts}")
    if end_ts != expected_end_ts:
        raise ValueError(f"End timestamp mismatch: got {end_ts}, expected {expected_end_ts}")
        
    print(f"   [Self-Check Passed]: {csv_path.name}")
    print(f"      Rows: {len(df_read):,} | Missing Targets: {actual_nan_targets} | NaN Residuals: {actual_nan_res}")
    print(f"      Timespan: {start_ts}  -->  {end_ts}")


def main():
    print("=" * 75)
    print("ELECTRICITY CONSUMPTION FORECASTING & ANOMALY DETECTION")
    print("Milestone 1: Validation & Test Forecast / Residual Generation")
    print("=" * 75)
    print(f"Project root resolved: {PROJECT_ROOT}")
    
    # 1. Load Dataset
    df = load_dataset(FEATURES_PATH)
    
    # 2. Load Split Specifications
    splits_meta = load_split_specification(SPLIT_JSON_PATH)
    val_meta = splits_meta["validation"]
    test_meta = splits_meta["test"]
    
    # 3. Extract exact validation and test slices
    val_slice = df.iloc[val_meta["start_idx"] : val_meta["end_idx"] + 1].copy()
    test_slice = df.iloc[test_meta["start_idx"] : test_meta["end_idx"] + 1].copy()
    
    # 4. Load XGBoost Model and verify features
    model, best_iter = load_and_verify_model(XGB_MODEL_PATH)
    
    # 5. Generate Predictions and Residuals
    val_predictions_df = generate_forecasts_and_residuals(
        val_slice, "Validation", model, best_iter
    )
    test_predictions_df = generate_forecasts_and_residuals(
        test_slice, "Test", model, best_iter
    )
    
    # 6. Ensure Output Directory Exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 7. Save Prediction CSVs
    print("\nSaving forecast and residual records...")
    val_predictions_df.to_csv(VAL_PREDICTIONS_PATH, index=False)
    print(f"   Saved: {VAL_PREDICTIONS_PATH} ({VAL_PREDICTIONS_PATH.stat().st_size:,} bytes)")
    
    test_predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)
    print(f"   Saved: {TEST_PREDICTIONS_PATH} ({TEST_PREDICTIONS_PATH.stat().st_size:,} bytes)")
    
    # 8. Strict Verification of Output Files
    print("\n" + "=" * 75)
    print("STRICT SELF-VERIFICATION OF SAVED ARTIFACTS")
    print("=" * 75)
    
    verify_output_file(
        VAL_PREDICTIONS_PATH,
        expected_rows=5778,
        expected_nan_targets=22,
        expected_start_ts="2016-09-06 12:00:00+05:30",
        expected_end_ts="2017-05-05 05:00:00+05:30"
    )
    
    verify_output_file(
        TEST_PREDICTIONS_PATH,
        expected_rows=5778,
        expected_nan_targets=118,
        expected_start_ts="2017-05-05 06:00:00+05:30",
        expected_end_ts="2017-12-31 23:00:00+05:30"
    )
    
    print("\n" + "=" * 75)
    print("MILESTONE 1 COMPLETE: Forecast and residual generation successful.")
    print("=" * 75)


if __name__ == "__main__":
    main()
