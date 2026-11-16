"""
Electricity Consumption Forecasting & Anomaly Detection Pipeline
Component: Isolation Forest-Based Anomaly Detection
File: project/src/anomaly_detection/isolation_forest_detection.py

Responsibilities:
1. Resolve project paths robustly using Path(__file__).resolve().parents[2].
2. Load feature dataset (academic_features.csv) and split specification (split_specification.json).
3. Validate exact 39 approved predictive features in exact sequence (strictly no target/leakage).
4. Extract chronological splits:
   - Training: indices 0 to 26963 (26,964 rows)
   - Validation: indices 26964 to 32741 (5,778 rows)
   - Test: indices 32742 to 38519 (5,778 rows)
5. Preprocess features with median imputation (SimpleImputer fitted strictly on training data).
6. Save fitted imputer to models/anomaly_imputer.joblib.
7. Fit Isolation Forest (n_estimators=300, random_state=42, contamination='auto', n_jobs=-1)
   strictly on training features.
8. Save fitted model to models/isolation_forest.joblib.
9. Compute decision functions and anomaly scores:
   anomaly_score = -decision_function (higher score consistently means more anomalous).
10. Calibrate anomaly threshold on validation scores only (95th percentile operating point).
11. Freeze validation threshold and evaluate validation and test sets.
12. Save validation_isolation_forest.csv and test_isolation_forest.csv (preserving all 5,778 rows).
13. Save reports/anomaly_detection/isolation_forest_thresholds.json and isolation_forest_summary.csv.
14. Generate 4 professional Matplotlib visualizations (200 DPI PNGs).
15. Perform strict self-verification against all project requirements.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Path Configuration (Dynamically resolved relative to project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project root: project/

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "academic_features.csv"
SPLIT_JSON_PATH = PROJECT_ROOT / "reports" / "split_specification.json"

MODELS_DIR = PROJECT_ROOT / "models"
IMPUTER_SAVE_PATH = MODELS_DIR / "anomaly_imputer.joblib"
ISO_FOREST_SAVE_PATH = MODELS_DIR / "isolation_forest.joblib"

DATA_ANOMALY_DIR = PROJECT_ROOT / "data" / "anomaly_detection"
REPORTS_ANOMALY_DIR = PROJECT_ROOT / "reports" / "anomaly_detection"
VIZ_DIR = REPORTS_ANOMALY_DIR / "visualizations"

VAL_OUT_PATH = DATA_ANOMALY_DIR / "validation_isolation_forest.csv"
TEST_OUT_PATH = DATA_ANOMALY_DIR / "test_isolation_forest.csv"
SUMMARY_CSV_PATH = REPORTS_ANOMALY_DIR / "isolation_forest_summary.csv"
THRESHOLDS_JSON_PATH = REPORTS_ANOMALY_DIR / "isolation_forest_thresholds.json"

# Milestone 2 files for timestamp alignment verification
VAL_M2_PATH = DATA_ANOMALY_DIR / "validation_anomalies.csv"
TEST_M2_PATH = DATA_ANOMALY_DIR / "test_anomalies.csv"

# ---------------------------------------------------------------------------
# Exact 39 Approved Predictive Features (Order Preserved)
# Strictly NO target, alias, contemporaneous electrical, residual, or anomaly flags
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

EXCLUDED_COLS: List[str] = [
    "timestamp", "power", "power_mean", "power_min", "power_max", "power_std",
    "power_range", "coefficient_of_variation", "energy_kwh", "current",
    "voltage", "frequency", "power_factor"
]

RANDOM_STATE = 42
N_ESTIMATORS = 300
CONTAMINATION = "auto"
OPERATING_PERCENTILE = 95.0


def load_dataset_and_splits() -> Tuple[pd.DataFrame, Dict]:
    """Load dataset and extract chronological split boundaries."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found: {FEATURES_PATH}")
    if not SPLIT_JSON_PATH.exists():
        raise FileNotFoundError(f"Split specification not found: {SPLIT_JSON_PATH}")
        
    print(f"Loading feature dataset: {FEATURES_PATH.name}")
    df = pd.read_csv(FEATURES_PATH)
    print(f"   Total shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]
        
    return df, splits


def validate_features(df: pd.DataFrame) -> None:
    """Strictly validate feature list against exclusions and dataset columns."""
    if len(FEATURES) != 39:
        raise ValueError(f"Expected exactly 39 approved features, got {len(FEATURES)}")
        
    for feat in FEATURES:
        if feat not in df.columns:
            raise KeyError(f"Approved feature '{feat}' missing from dataset!")
        if feat in EXCLUDED_COLS:
            raise ValueError(f"LEAKAGE ALERT: Feature '{feat}' is in EXCLUDED_COLS!")
            
    print("   [Feature Verification]: SUCCESS! Exactly 39 predictive features verified. Zero leakage.")


def fit_imputer_and_transform(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer]:
    """
    Fit SimpleImputer(strategy='median') strictly on training features.
    Transform train, validation, and test sets.
    """
    print("\n" + "=" * 75)
    print("1. FEATURE IMPUTATION (STRICT TRAINING-ONLY FITTING)")
    print("=" * 75)
    
    # Check for all-NaN columns in training
    train_nans = X_train.isna().sum()
    all_nan_cols = train_nans[train_nans == len(X_train)].index.tolist()
    if all_nan_cols:
        raise ValueError(f"All-NaN column detected in training features: {all_nan_cols}")
        
    print(f"   Fitting SimpleImputer(strategy='median') on {len(X_train):,} training rows...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    
    print("   Transforming validation and test features using fitted training imputer...")
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    
    # Assert matrices are 100% complete and finite
    assert np.all(np.isfinite(X_train_imp)), "NaN/Inf in imputed training features"
    assert np.all(np.isfinite(X_val_imp)), "NaN/Inf in imputed validation features"
    assert np.all(np.isfinite(X_test_imp)), "NaN/Inf in imputed test features"
    
    print(f"   [Imputation Check]: SUCCESS! All matrices finite. Train={X_train_imp.shape}, Val={X_val_imp.shape}, Test={X_test_imp.shape}")
    
    # Save fitted imputer
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, IMPUTER_SAVE_PATH)
    print(f"   Saved fitted imputer -> {IMPUTER_SAVE_PATH.name} ({IMPUTER_SAVE_PATH.stat().st_size:,} bytes)")
    
    return X_train_imp, X_val_imp, X_test_imp, imputer


def fit_isolation_forest(X_train_imp: np.ndarray) -> IsolationForest:
    """Fit Isolation Forest model strictly on training features."""
    print("\n" + "=" * 75)
    print("2. TRAINING ISOLATION FOREST MODEL (STRICT TRAINING-ONLY)")
    print("=" * 75)
    print(f"   Configuration: n_estimators={N_ESTIMATORS}, random_state={RANDOM_STATE}, contamination='{CONTAMINATION}', n_jobs=-1")
    
    t0 = time.time()
    iso_forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        contamination=CONTAMINATION,
        n_jobs=-1
    )
    iso_forest.fit(X_train_imp)
    train_time = time.time() - t0
    
    print(f"   Isolation Forest training completed in {train_time:.2f} seconds.")
    
    # Save fitted model
    joblib.dump(iso_forest, ISO_FOREST_SAVE_PATH)
    print(f"   Saved fitted model -> {ISO_FOREST_SAVE_PATH.name} ({ISO_FOREST_SAVE_PATH.stat().st_size:,} bytes)")
    
    return iso_forest


def compute_scores_and_calibrate(
    iso_forest: IsolationForest,
    X_val_imp: np.ndarray,
    X_test_imp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute decision functions and anomaly scores:
        anomaly_score = -decision_function
        (Higher score consistently indicates higher degree of anomaly)
        
    Calibrate threshold strictly on validation scores using 95th percentile.
    Freeze threshold before applying to test set.
    """
    print("\n" + "=" * 75)
    print("3. ANOMALY SCORING & VALIDATION-ONLY THRESHOLD CALIBRATION")
    print("=" * 75)
    
    # Decision functions (scikit-learn: lower means more abnormal)
    val_dec = iso_forest.decision_function(X_val_imp)
    test_dec = iso_forest.decision_function(X_test_imp)
    
    # Anomaly scores: inverted so higher means more anomalous
    val_score = -val_dec
    test_score = -test_dec
    
    # Calibrate on validation scores only
    val_threshold = float(np.percentile(val_score, OPERATING_PERCENTILE))
    print(f"   Validation Anomaly Score Range: [{val_score.min():.6f}, {val_score.max():.6f}]")
    print(f"   Calibrated {OPERATING_PERCENTILE}th Percentile Threshold: {val_threshold:.6f}")
    print(f"   Threshold Rule: isolation_forest_anomaly = (anomaly_score >= {val_threshold:.6f})")
    
    val_anomalies = (val_score >= val_threshold)
    test_anomalies = (test_score >= val_threshold)
    
    val_anom_count = int(val_anomalies.sum())
    val_pct = (val_anom_count / len(val_anomalies)) * 100.0
    print(f"   Validation Anomalies: {val_anom_count} / {len(val_anomalies):,} ({val_pct:.2f}%)")
    
    test_anom_count = int(test_anomalies.sum())
    test_pct = (test_anom_count / len(test_anomalies)) * 100.0
    print(f"   Test Anomalies (Frozen Threshold): {test_anom_count} / {len(test_anomalies):,} ({test_pct:.2f}%)")
    
    return val_dec, val_score, test_dec, test_score, val_threshold


# ---------------------------------------------------------------------------
# Visualizations (Professional Matplotlib Plots - 200 DPI)
# ---------------------------------------------------------------------------

def plot_score_distribution(
    scores: np.ndarray,
    threshold: float,
    split_name: str,
    output_path: Path,
    bar_color: str
) -> None:
    """Plot anomaly score distribution with highlighted anomaly tail and threshold."""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    counts, bins, patches = ax.hist(
        scores, bins=65, edgecolor="#222222", linewidth=0.6,
        alpha=0.75, color=bar_color, label=f"{split_name} Anomaly Scores"
    )
    
    # Highlight upper anomaly tail
    for patch, left_edge, right_edge in zip(patches, bins[:-1], bins[1:]):
        if right_edge >= threshold:
            patch.set_facecolor("#d62728")
            patch.set_alpha(0.9)
            
    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=2.0,
               label=f"Frozen Validation Threshold ({threshold:.4f})")
    ax.axvline(float(np.median(scores)), color="#2ca02c", linestyle="-", linewidth=1.5,
               label=f"Median Score ({np.median(scores):.4f})")
               
    ax.set_title(f"Isolation Forest Anomaly Score Distribution ({split_name} Split)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Isolation Forest Anomaly Score (-decision_function, Higher = More Anomalous)",
                  fontsize=10.5, fontweight="semibold")
    ax.set_ylabel("Observation Frequency (Hours)", fontsize=10.5, fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
    
    anom_count = int((scores >= threshold).sum())
    stats_text = (
        f"{split_name} Total N: {len(scores):,}\n"
        f"Mean Score: {np.mean(scores):.4f}\n"
        f"Median Score: {np.median(scores):.4f}\n"
        f"Std Score: {np.std(scores):.4f}\n"
        f"Anomalies: {anom_count:,} ({anom_count/len(scores)*100:.2f}%)"
    )
    ax.text(0.03, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            fontsize=9.5, family="monospace")
            
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot Saved]: Score Distribution ({split_name}) -> {output_path.name}")


def plot_timeseries_anomalies(
    timestamps: pd.Series,
    scores: np.ndarray,
    threshold: float,
    split_name: str,
    output_path: Path
) -> None:
    """Plot chronological anomaly scores with detected anomalies."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ts = pd.to_datetime(timestamps)
    anom_mask = (scores >= threshold)
    norm_mask = ~anom_mask
    
    # Baseline line
    ax.plot(ts, scores, color="#7f7f7f", linewidth=0.8, alpha=0.5, label="Anomaly Score")
    
    # Normal points
    ax.scatter(ts[norm_mask], scores[norm_mask], color="#1f77b4", s=10, alpha=0.4, label="Normal Observations")
    
    # Anomalies
    if anom_mask.sum() > 0:
        ax.scatter(ts[anom_mask], scores[anom_mask], color="#d62728", s=32, marker="o",
                   label=f"Isolation Forest Anomalies (N={anom_mask.sum():,})", zorder=5)
                   
    ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.8,
               label=f"Frozen Validation Threshold ({threshold:.4f})")
    ax.axhline(float(np.median(scores)), color="#2ca02c", linestyle=":", linewidth=1.2,
               label=f"Median Score ({np.median(scores):.4f})")
               
    ax.set_title(f"{split_name} Split: Isolation Forest Anomaly Timeline",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Timestamp", fontsize=10.5, fontweight="semibold")
    ax.set_ylabel("Anomaly Score (-decision_function)", fontsize=10.5, fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot Saved]: Anomaly Timeline ({split_name}) -> {output_path.name}")


# ---------------------------------------------------------------------------
# Strict Verification Functions
# ---------------------------------------------------------------------------

def verify_output_datasets(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_m2_path: Path,
    test_m2_path: Path
) -> None:
    """Verify row counts, timestamps, finite scores, and schema against requirements."""
    print("\n" + "=" * 75)
    print("STRICT SELF-VERIFICATION OF OUTPUT DATASETS")
    print("=" * 75)
    
    expected_cols = [
        "timestamp",
        "isolation_forest_decision_function",
        "isolation_forest_anomaly_score",
        "isolation_forest_anomaly"
    ]
    
    for df, name, m2_path in [(val_df, "Validation", val_m2_path), (test_df, "Test", test_m2_path)]:
        # Check rows
        if len(df) != 5778:
            raise ValueError(f"Row count mismatch in {name} output: {len(df)} vs 5,778")
            
        # Check columns
        if list(df.columns) != expected_cols:
            raise ValueError(f"Column schema mismatch in {name} output: {list(df.columns)} vs {expected_cols}")
            
        # Check timestamps against Milestone 2
        m2_df = pd.read_csv(m2_path)
        if not (df["timestamp"].values == m2_df["timestamp"].values).all():
            raise ValueError(f"Timestamp sequence mismatch between Isolation Forest {name} and Milestone 2!")
            
        # Check finite scores
        if not np.all(np.isfinite(df["isolation_forest_decision_function"])):
            raise ValueError(f"Non-finite values in decision function of {name} output!")
        if not np.all(np.isfinite(df["isolation_forest_anomaly_score"])):
            raise ValueError(f"Non-finite values in anomaly score of {name} output!")
            
        # Check score inversion
        expected_score = -df["isolation_forest_decision_function"]
        diff = np.max(np.abs(df["isolation_forest_anomaly_score"] - expected_score))
        if diff > 1e-5:
            raise ValueError(f"Score inversion mismatch in {name} output: max diff = {diff}")
            
        anom_count = int((df["isolation_forest_anomaly"] == True).sum())
        anom_pct = (anom_count / len(df)) * 100.0
        print(f"   [Self-Check Passed]: {name} Output File ({len(df):,} rows)")
        print(f"      Timestamps exact match Milestone 2: True")
        print(f"      All scores finite: True | Anomaly Count: {anom_count:,} ({anom_pct:.2f}%)")


def main():
    print("=" * 75)
    print("ELECTRICITY CONSUMPTION FORECASTING & ANOMALY DETECTION")
    print("Milestone 3: Isolation Forest-Based Anomaly Detection Pipeline")
    print("=" * 75)
    print(f"Project root resolved: {PROJECT_ROOT}")
    
    # 1. Load Data and Split Specification
    df, splits = load_dataset_and_splits()
    validate_features(df)
    
    train_meta = splits["train"]
    val_meta = splits["validation"]
    test_meta = splits["test"]
    
    train_df = df.iloc[train_meta["start_idx"] : train_meta["end_idx"] + 1].copy()
    val_df = df.iloc[val_meta["start_idx"] : val_meta["end_idx"] + 1].copy()
    test_df = df.iloc[test_meta["start_idx"] : test_meta["end_idx"] + 1].copy()
    
    print(f"   Train split:      {len(train_df):,} rows (indices {train_meta['start_idx']} to {train_meta['end_idx']})")
    print(f"   Validation split: {len(val_df):,} rows (indices {val_meta['start_idx']} to {val_meta['end_idx']})")
    print(f"   Test split:       {len(test_df):,} rows (indices {test_meta['start_idx']} to {test_meta['end_idx']})")
    
    X_train = train_df[FEATURES]
    X_val = val_df[FEATURES]
    X_test = test_df[FEATURES]
    
    # 2. Imputation (Training Only)
    X_train_imp, X_val_imp, X_test_imp, imputer = fit_imputer_and_transform(X_train, X_val, X_test)
    
    # 3. Model Training (Training Only)
    iso_forest = fit_isolation_forest(X_train_imp)
    
    # 4. Scoring and Validation Calibration
    val_dec, val_score, test_dec, test_score, val_th = compute_scores_and_calibrate(
        iso_forest, X_val_imp, X_test_imp
    )
    
    # 5. Build Output DataFrames
    val_out_df = pd.DataFrame({
        "timestamp": val_df["timestamp"].values,
        "isolation_forest_decision_function": np.round(val_dec, 6),
        "isolation_forest_anomaly_score": np.round(val_score, 6),
        "isolation_forest_anomaly": (val_score >= val_th)
    })
    
    test_out_df = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "isolation_forest_decision_function": np.round(test_dec, 6),
        "isolation_forest_anomaly_score": np.round(test_score, 6),
        "isolation_forest_anomaly": (test_score >= val_th)
    })
    
    # 6. Save Output Datasets
    print("\n" + "=" * 75)
    print("4. SAVING OUTPUT DATASETS & SUMMARY REPORTS")
    print("=" * 75)
    DATA_ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    val_out_df.to_csv(VAL_OUT_PATH, index=False)
    print(f"   Saved: {VAL_OUT_PATH.name} ({VAL_OUT_PATH.stat().st_size:,} bytes)")
    
    test_out_df.to_csv(TEST_OUT_PATH, index=False)
    print(f"   Saved: {TEST_OUT_PATH.name} ({TEST_OUT_PATH.stat().st_size:,} bytes)")
    
    # 7. Save Summary CSV
    summary_records = [
        {
            "Split": "Validation",
            "Total_Rows": len(val_out_df),
            "Scored_Rows": int(np.isfinite(val_score).sum()),
            "Unavailable_Score_Rows": int((~np.isfinite(val_score)).sum()),
            "Anomaly_Count": int((val_out_df["isolation_forest_anomaly"] == True).sum()),
            "Non_Anomaly_Count": int((val_out_df["isolation_forest_anomaly"] == False).sum()),
            "Anomaly_Percentage": round(float((val_out_df["isolation_forest_anomaly"] == True).mean() * 100.0), 2),
            "Score_Min": round(float(np.min(val_score)), 6),
            "Score_Max": round(float(np.max(val_score)), 6),
            "Score_Mean": round(float(np.mean(val_score)), 6),
            "Score_Median": round(float(np.median(val_score)), 6),
            "Score_Std": round(float(np.std(val_score)), 6)
        },
        {
            "Split": "Test",
            "Total_Rows": len(test_out_df),
            "Scored_Rows": int(np.isfinite(test_score).sum()),
            "Unavailable_Score_Rows": int((~np.isfinite(test_score)).sum()),
            "Anomaly_Count": int((test_out_df["isolation_forest_anomaly"] == True).sum()),
            "Non_Anomaly_Count": int((test_out_df["isolation_forest_anomaly"] == False).sum()),
            "Anomaly_Percentage": round(float((test_out_df["isolation_forest_anomaly"] == True).mean() * 100.0), 2),
            "Score_Min": round(float(np.min(test_score)), 6),
            "Score_Max": round(float(np.max(test_score)), 6),
            "Score_Mean": round(float(np.mean(test_score)), 6),
            "Score_Median": round(float(np.median(test_score)), 6),
            "Score_Std": round(float(np.std(test_score)), 6)
        }
    ]
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"   Saved: {SUMMARY_CSV_PATH.name} ({SUMMARY_CSV_PATH.stat().st_size:,} bytes)")
    print("\n=== ISOLATION FOREST PERFORMANCE SUMMARY ===")
    print(summary_df.to_string(index=False))
    
    # 8. Save Metadata JSON
    metadata = {
        "calibration_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_configuration": {
            "model_type": "IsolationForest",
            "n_estimators": N_ESTIMATORS,
            "max_samples": "auto",
            "contamination": CONTAMINATION,
            "max_features": 1.0,
            "bootstrap": False,
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        },
        "random_seed": RANDOM_STATE,
        "feature_count": len(FEATURES),
        "feature_names": FEATURES,
        "split_row_counts": {
            "training_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df)
        },
        "imputation_strategy": {
            "strategy": "median",
            "imputer_type": "SimpleImputer",
            "fitted_on": f"training_features_only (indices {train_meta['start_idx']} to {train_meta['end_idx']})",
            "saved_path": str(IMPUTER_SAVE_PATH.relative_to(PROJECT_ROOT))
        },
        "training_isolation_forest_confirmation": {
            "fitted_on": f"training_features_only (indices {train_meta['start_idx']} to {train_meta['end_idx']})",
            "saved_path": str(ISO_FOREST_SAVE_PATH.relative_to(PROJECT_ROOT)),
            "validation_or_test_leakage": False
        },
        "threshold_calibration": {
            "calibration_split": "validation_only",
            "calibrated_percentile": OPERATING_PERCENTILE,
            "validation_threshold": round(val_th, 6),
            "score_direction": "Higher score indicates higher anomaly likelihood: anomaly_score = -decision_function",
            "threshold_rule": f"isolation_forest_anomaly = (isolation_forest_anomaly_score >= {val_th:.6f})",
            "threshold_rationale": (
                f"{OPERATING_PERCENTILE}th percentile of validation anomaly scores used as experimental "
                f"operating point (flags top {100.0 - OPERATING_PERCENTILE:.1f}% most isolated validation observations, "
                "frozen strictly before evaluating test set). Not a verified true anomaly rate."
            )
        },
        "evaluation_summary": {
            "validation": {
                "total_rows": len(val_out_df),
                "scored_rows": int(np.isfinite(val_score).sum()),
                "unavailable_score_rows": 0,
                "anomaly_count": int((val_out_df["isolation_forest_anomaly"] == True).sum()),
                "anomaly_percentage": round(float((val_out_df["isolation_forest_anomaly"] == True).mean() * 100.0), 2)
            },
            "test": {
                "total_rows": len(test_out_df),
                "scored_rows": int(np.isfinite(test_score).sum()),
                "unavailable_score_rows": 0,
                "anomaly_count": int((test_out_df["isolation_forest_anomaly"] == True).sum()),
                "anomaly_percentage": round(float((test_out_df["isolation_forest_anomaly"] == True).mean() * 100.0), 2)
            }
        }
    }
    
    with open(THRESHOLDS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"   Saved: {THRESHOLDS_JSON_PATH.name} ({THRESHOLDS_JSON_PATH.stat().st_size:,} bytes)")
    
    # 9. Generate Visualizations
    print("\n" + "=" * 75)
    print("5. GENERATING VISUALIZATIONS (REPORTS/ANOMALY_DETECTION/VISUALIZATIONS)")
    print("=" * 75)
    
    # Plot 1: Validation score distribution
    plot_score_distribution(
        val_score, val_th, "Validation",
        VIZ_DIR / "isolation_forest_validation_score_distribution.png",
        bar_color="#1f77b4"
    )
    
    # Plot 2: Test score distribution
    plot_score_distribution(
        test_score, val_th, "Test",
        VIZ_DIR / "isolation_forest_test_score_distribution.png",
        bar_color="#ff7f0e"
    )
    
    # Plot 3: Validation timeline
    plot_timeseries_anomalies(
        val_out_df["timestamp"], val_score, val_th, "Validation",
        VIZ_DIR / "isolation_forest_validation_timeseries.png"
    )
    
    # Plot 4: Test timeline
    plot_timeseries_anomalies(
        test_out_df["timestamp"], test_score, val_th, "Test",
        VIZ_DIR / "isolation_forest_test_timeseries.png"
    )
    
    # 10. Strict Verification
    verify_output_datasets(val_out_df, test_out_df, VAL_M2_PATH, TEST_M2_PATH)
    
    print("\n" + "=" * 75)
    print("MILESTONE 3 COMPLETE: Isolation Forest Detection Successful.")
    print("=" * 75)


if __name__ == "__main__":
    main()
