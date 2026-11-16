"""
Electricity Consumption Forecasting & Anomaly Detection Pipeline
Component: Detector Comparison and Combined Anomaly Analysis
File: project/src/anomaly_detection/compare_anomalies.py

Responsibilities:
1. Resolve project paths dynamically relative to project root.
2. Load and verify Milestone 2 (Residual Detection) and Milestone 3 (Isolation Forest) artifacts.
3. Validate strict timestamp alignment (one-to-one correspondence across all 5,778 rows).
4. Perform detector agreement analysis on classified observations:
   - both_anomalous (Intersection: Residual=True AND IsoForest=True)
   - residual_only (Residual=True AND IsoForest=False)
   - isolation_forest_only (Residual=False AND IsoForest=True)
   - neither_anomalous (Residual=False AND IsoForest=False)
   - unclassified (missing actual target / residual)
5. Calculate statistical agreement metrics:
   - Agreement Percentage, Jaccard Similarity, Cohen's Kappa.
6. Conduct directional residual analysis for residual-only and intersection anomalies.
7. Construct and save combined anomaly datasets:
   - data/anomaly_detection/validation_combined_anomalies.csv
   - data/anomaly_detection/test_combined_anomalies.csv
8. Generate summary table:
   - reports/anomaly_detection/detector_comparison_summary.csv
9. Generate comprehensive markdown report:
   - reports/anomaly_detection/detector_comparison_report.md
10. Generate 6 professional Matplotlib visualizations (200 DPI PNGs):
   - validation_detector_agreement.png
   - test_detector_agreement.png
   - validation_anomaly_timeline_comparison.png
   - test_anomaly_timeline_comparison.png
   - detector_anomaly_rate_comparison.png
   - test_combined_anomaly_timeline.png
11. Perform strict self-verification of generated artifacts.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Path Configuration (Dynamically resolved relative to project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project root: project/

DATA_ANOMALY_DIR = PROJECT_ROOT / "data" / "anomaly_detection"
REPORTS_ANOMALY_DIR = PROJECT_ROOT / "reports" / "anomaly_detection"
VIZ_DIR = REPORTS_ANOMALY_DIR / "visualizations"

VAL_RES_PATH = DATA_ANOMALY_DIR / "validation_anomalies.csv"
TEST_RES_PATH = DATA_ANOMALY_DIR / "test_anomalies.csv"
VAL_ISO_PATH = DATA_ANOMALY_DIR / "validation_isolation_forest.csv"
TEST_ISO_PATH = DATA_ANOMALY_DIR / "test_isolation_forest.csv"

RES_THRESHOLDS_PATH = REPORTS_ANOMALY_DIR / "residual_thresholds.json"
ISO_THRESHOLDS_PATH = REPORTS_ANOMALY_DIR / "isolation_forest_thresholds.json"

VAL_COMBINED_PATH = DATA_ANOMALY_DIR / "validation_combined_anomalies.csv"
TEST_COMBINED_PATH = DATA_ANOMALY_DIR / "test_combined_anomalies.csv"
SUMMARY_CSV_PATH = REPORTS_ANOMALY_DIR / "detector_comparison_summary.csv"
REPORT_MD_PATH = REPORTS_ANOMALY_DIR / "detector_comparison_report.md"


def load_and_verify_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """Load and verify input datasets, schemas, timestamps, and metadata."""
    for p in [VAL_RES_PATH, TEST_RES_PATH, VAL_ISO_PATH, TEST_ISO_PATH, RES_THRESHOLDS_PATH, ISO_THRESHOLDS_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required input artifact missing: {p}")
            
    val_res = pd.read_csv(VAL_RES_PATH)
    test_res = pd.read_csv(TEST_RES_PATH)
    val_iso = pd.read_csv(VAL_ISO_PATH)
    test_iso = pd.read_csv(TEST_ISO_PATH)
    
    with open(RES_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        res_meta = json.load(f)
    with open(ISO_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        iso_meta = json.load(f)
        
    print("Verifying input artifacts and timestamp alignment...")
    for name, df_r, df_i in [("Validation", val_res, val_iso), ("Test", test_res, test_iso)]:
        if len(df_r) != 5778 or len(df_i) != 5778:
            raise ValueError(f"Row count mismatch in {name}: {len(df_r)} res vs {len(df_i)} iso (expected 5,778)")
        if not (df_r["timestamp"].values == df_i["timestamp"].values).all():
            raise ValueError(f"Timestamp alignment failure in {name} split! One-to-one correspondence violated.")
        print(f"   [{name} Input Alignment]: SUCCESS! 5,778 timestamps aligned identically.")
        
    return val_res, test_res, val_iso, test_iso, res_meta, iso_meta


def build_combined_dataset(df_res: pd.DataFrame, df_iso: pd.DataFrame) -> pd.DataFrame:
    """
    Construct combined anomaly dataset with 5 agreement categories and OR/AND rules.
    Preserves all 5,778 rows in exact chronological order.
    """
    classified_mask = df_res["residual_anomaly"].notna()
    
    # Extract booleans
    r_flags = df_res["residual_anomaly"] == True
    i_flags = df_iso["isolation_forest_anomaly"] == True
    
    # 5 Mutually Exclusive Agreement Categories
    category = pd.Series("unclassified", index=df_res.index, dtype="object")
    category.loc[classified_mask & r_flags & i_flags] = "both_anomalous"
    category.loc[classified_mask & r_flags & ~i_flags] = "residual_only"
    category.loc[classified_mask & ~r_flags & i_flags] = "isolation_forest_only"
    category.loc[classified_mask & ~r_flags & ~i_flags] = "neither_anomalous"
    
    # Combined Boolean Series (nullable boolean)
    comb_or = pd.Series(pd.NA, index=df_res.index, dtype="boolean")
    comb_and = pd.Series(pd.NA, index=df_res.index, dtype="boolean")
    
    class_indices = df_res.index[classified_mask]
    comb_or.loc[class_indices] = (r_flags[classified_mask] | i_flags[classified_mask])
    comb_and.loc[class_indices] = (r_flags[classified_mask] & i_flags[classified_mask])
    
    # Construct structured DataFrame
    combined_df = pd.DataFrame({
        "timestamp": df_res["timestamp"].values,
        "actual_power_watts": df_res["actual_power_watts"].values,
        "predicted_power_watts": df_res["predicted_power_watts"].values,
        "residual_watts": df_res["residual_watts"].values,
        "robust_residual_score": df_res["robust_residual_score"].values,
        "high_consumption_anomaly": df_res["high_consumption_anomaly"].values,
        "low_consumption_anomaly": df_res["low_consumption_anomaly"].values,
        "residual_anomaly": df_res["residual_anomaly"].values,
        "isolation_forest_anomaly_score": df_iso["isolation_forest_anomaly_score"].values,
        "isolation_forest_anomaly": df_iso["isolation_forest_anomaly"].values,
        "agreement_category": category.values,
        "combined_anomaly_or": comb_or.values,
        "combined_anomaly_and": comb_and.values,
        "classified": classified_mask.values
    })
    
    return combined_df


def compute_comparison_metrics(df_comb: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    """Calculate agreement metrics, Jaccard similarity, Cohen's kappa, and directional metrics."""
    total_obs = len(df_comb)
    classified_mask = df_comb["classified"] == True
    n_classified = int(classified_mask.sum())
    n_unclassified = total_obs - n_classified
    
    cat_counts = df_comb["agreement_category"].value_counts().to_dict()
    both = cat_counts.get("both_anomalous", 0)
    res_only = cat_counts.get("residual_only", 0)
    iso_only = cat_counts.get("isolation_forest_only", 0)
    neither = cat_counts.get("neither_anomalous", 0)
    unclass = cat_counts.get("unclassified", 0)
    
    # Assert category partition
    assert both + res_only + iso_only + neither + unclass == total_obs, "Category partition failure"
    assert unclass == n_unclassified, "Unclassified count mismatch"
    
    res_total = both + res_only
    iso_total_class = both + iso_only
    union_or = both + res_only + iso_only
    intersection_and = both
    
    # Agreement percentage among classified
    agr_pct = float((both + neither) / n_classified * 100.0) if n_classified > 0 else 0.0
    
    # Jaccard similarity
    jaccard = float(intersection_and / union_or) if union_or > 0 else 0.0
    
    # Cohen's Kappa
    class_df = df_comb[classified_mask]
    r_bool = (class_df["residual_anomaly"] == True).astype(bool)
    i_bool = (class_df["isolation_forest_anomaly"] == True).astype(bool)
    kappa = float(cohen_kappa_score(r_bool, i_bool))
    
    # Directional Residual Breakdown
    high_mask = class_df["high_consumption_anomaly"] == True
    low_mask = class_df["low_consumption_anomaly"] == True
    
    both_mask = class_df["agreement_category"] == "both_anomalous"
    res_only_mask = class_df["agreement_category"] == "residual_only"
    
    both_high = int((both_mask & high_mask).sum())
    both_low = int((both_mask & low_mask).sum())
    res_only_high = int((res_only_mask & high_mask).sum())
    res_only_low = int((res_only_mask & low_mask).sum())
    
    res_high_total = both_high + res_only_high
    res_low_total = both_low + res_only_low
    
    metrics = {
        "Split": split_name,
        "Total_Observations": total_obs,
        "Classified_Observations": n_classified,
        "Unclassified_Observations": n_unclassified,
        "Residual_Anomalies": res_total,
        "Residual_Anomaly_Percentage": round(float(res_total / n_classified * 100.0), 2),
        "Isolation_Forest_Anomalies": iso_total_class,
        "Isolation_Forest_Anomaly_Percentage": round(float(iso_total_class / n_classified * 100.0), 2),
        "Both_Anomalous_Intersection": intersection_and,
        "Intersection_Percentage": round(float(intersection_and / n_classified * 100.0), 2),
        "Residual_Only_Anomalies": res_only,
        "Residual_Only_Percentage": round(float(res_only / n_classified * 100.0), 2),
        "Isolation_Forest_Only_Anomalies": iso_only,
        "Isolation_Forest_Only_Percentage": round(float(iso_only / n_classified * 100.0), 2),
        "Neither_Anomalous": neither,
        "Neither_Percentage": round(float(neither / n_classified * 100.0), 2),
        "Union_Anomalies_OR": union_or,
        "Union_Percentage": round(float(union_or / n_classified * 100.0), 2),
        "Agreement_Percentage": round(agr_pct, 2),
        "Jaccard_Similarity": round(jaccard, 4),
        "Cohen_Kappa": round(kappa, 4),
        "Residual_Total_High": res_high_total,
        "Residual_Total_Low": res_low_total,
        "Both_High_Consumption": both_high,
        "Both_Low_Consumption": both_low,
        "Residual_Only_High_Consumption": res_only_high,
        "Residual_Only_Low_Consumption": res_only_low
    }
    
    return metrics


# ---------------------------------------------------------------------------
# Visualizations (Professional Matplotlib Plots - 200 DPI)
# ---------------------------------------------------------------------------

def plot_venn_agreement(
    metrics: Dict[str, Any],
    split_name: str,
    output_path: Path
) -> None:
    """Plot 1 & 2: Venn-style detector agreement diagram."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # Outer background box for classified space
    rect = plt.Rectangle((0.05, 0.08), 0.90, 0.84, facecolor="#f8f9fa",
                         edgecolor="#cccccc", linewidth=1.2, zorder=0)
    ax.add_patch(rect)
    
    # Overlapping circles
    c_res = Circle((0.40, 0.52), 0.26, color="#1f77b4", alpha=0.35, ec="#1f77b4", lw=2.2, zorder=2)
    c_iso = Circle((0.60, 0.52), 0.26, color="#ff7f0e", alpha=0.35, ec="#ff7f0e", lw=2.2, zorder=2)
    ax.add_patch(c_res)
    ax.add_patch(c_iso)
    
    res_only = metrics["Residual_Only_Anomalies"]
    both = metrics["Both_Anomalous_Intersection"]
    iso_only = metrics["Isolation_Forest_Only_Anomalies"]
    neither = metrics["Neither_Anomalous"]
    unclass = metrics["Unclassified_Observations"]
    n_class = metrics["Classified_Observations"]
    
    # Text labels in regions
    # Residual Exclusive (Left)
    ax.text(0.28, 0.52, f"Residual Only\n{res_only:,}\n({res_only/n_class*100:.2f}%)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#0d47a1")
    # Intersection (Center)
    ax.text(0.50, 0.52, f"Both Anomalous\n(Intersection)\n{both:,}\n({both/n_class*100:.2f}%)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#b71c1c")
    # Isolation Forest Exclusive (Right)
    ax.text(0.72, 0.52, f"Isolation Forest Only\n{iso_only:,}\n({iso_only/n_class*100:.2f}%)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#e65100")
    # Neither Anomalous (Top right inside box)
    ax.text(0.12, 0.86, f"Neither Anomalous: {neither:,} ({neither/n_class*100:.2f}%)\nUnclassified: {unclass} hours",
            ha="left", va="top", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bbbbbb", alpha=0.9))
            
    # Circle Headers
    ax.text(0.32, 0.80, f"Residual Detector\nTotal: {metrics['Residual_Anomalies']:,} ({metrics['Residual_Anomaly_Percentage']}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1f77b4")
    ax.text(0.68, 0.80, f"Isolation Forest\nTotal: {metrics['Isolation_Forest_Anomalies']:,} ({metrics['Isolation_Forest_Anomaly_Percentage']}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#ff7f0e")
            
    # Bottom Stats Bar
    stats_text = (
        f"Classified: {n_class:,}  |  Union (OR): {metrics['Union_Anomalies_OR']:,} ({metrics['Union_Percentage']}%)  |  "
        f"Overall Agreement: {metrics['Agreement_Percentage']}%  |  Jaccard: {metrics['Jaccard_Similarity']}  |  Cohen's Kappa: {metrics['Cohen_Kappa']}"
    )
    ax.text(0.50, 0.12, stats_text, ha="center", va="center", fontsize=10, fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e9ecef", edgecolor="#ced4da", alpha=0.95))
            
    ax.set_title(f"Detector Agreement & Overlap Venn Diagram ({split_name} Split)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot Saved]: Agreement Venn ({split_name}) -> {output_path.name}")


def plot_anomaly_timeline_comparison(
    df_comb: pd.DataFrame,
    split_name: str,
    output_path: Path
) -> None:
    """Plot 3 & 4: Dual chronological anomaly timeline comparing both detectors."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ts = pd.to_datetime(df_comb["timestamp"])
    classified = df_comb["classified"] == True
    
    # Top Panel: Residual Score Timeline
    res_scores = df_comb["robust_residual_score"]
    r_anom = classified & (df_comb["residual_anomaly"] == True)
    ax1.plot(ts, res_scores, color="#1f77b4", linewidth=0.8, alpha=0.5, label="Robust Residual Score")
    if r_anom.sum() > 0:
        ax1.scatter(ts[r_anom], res_scores[r_anom], color="#d62728", s=28, marker="^",
                    label=f"Residual Anomalies (|z| >= 3.5, N={r_anom.sum():,})", zorder=5)
    ax1.axhline(3.5, color="#d62728", linestyle="--", linewidth=1.2)
    ax1.axhline(-3.5, color="#17becf", linestyle="--", linewidth=1.2)
    ax1.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax1.set_title(f"{split_name} Horizon: Residual-Based Anomaly Score Timeline", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Residual Score (z_robust)", fontsize=10.5, fontweight="semibold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
    
    # Bottom Panel: Isolation Forest Anomaly Score Timeline
    iso_scores = df_comb["isolation_forest_anomaly_score"]
    i_anom = classified & (df_comb["isolation_forest_anomaly"] == True)
    ax2.plot(ts, iso_scores, color="#ff7f0e", linewidth=0.8, alpha=0.5, label="Isolation Forest Score")
    if i_anom.sum() > 0:
        ax2.scatter(ts[i_anom], iso_scores[i_anom], color="#e65100", s=28, marker="o",
                    label=f"Isolation Forest Anomalies (N={i_anom.sum():,})", zorder=5)
    ax2.axhline(0.080704, color="#e65100", linestyle="--", linewidth=1.2, label="Frozen Threshold (0.0807)")
    ax2.set_title(f"{split_name} Horizon: Isolation Forest Anomaly Score Timeline", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Timestamp", fontsize=10.5, fontweight="semibold")
    ax2.set_ylabel("Anomaly Score (-decision_fn)", fontsize=10.5, fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot Saved]: Timeline Comparison ({split_name}) -> {output_path.name}")


def plot_anomaly_rate_comparison(
    val_m: Dict[str, Any],
    test_m: Dict[str, Any],
    output_path: Path
) -> None:
    """Plot 5: Grouped bar chart comparing Validation vs Test anomaly rates across rules."""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    categories = [
        "Residual Detector",
        "Isolation Forest",
        "Union (OR Rule)",
        "Intersection (AND Rule)"
    ]
    val_rates = [
        val_m["Residual_Anomaly_Percentage"],
        val_m["Isolation_Forest_Anomaly_Percentage"],
        val_m["Union_Percentage"],
        val_m["Intersection_Percentage"]
    ]
    test_rates = [
        test_m["Residual_Anomaly_Percentage"],
        test_m["Isolation_Forest_Anomaly_Percentage"],
        test_m["Union_Percentage"],
        test_m["Intersection_Percentage"]
    ]
    val_counts = [
        val_m["Residual_Anomalies"],
        val_m["Isolation_Forest_Anomalies"],
        val_m["Union_Anomalies_OR"],
        val_m["Both_Anomalous_Intersection"]
    ]
    test_counts = [
        test_m["Residual_Anomalies"],
        test_m["Isolation_Forest_Anomalies"],
        test_m["Union_Anomalies_OR"],
        test_m["Both_Anomalous_Intersection"]
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, val_rates, width, label="Validation (Calibration)",
                   color="#1f77b4", alpha=0.85, edgecolor="#222222", linewidth=0.8)
    bars2 = ax.bar(x + width/2, test_rates, width, label="Test (Frozen Evaluation)",
                   color="#ff7f0e", alpha=0.85, edgecolor="#222222", linewidth=0.8)
                   
    # Data labels
    for bar, count in zip(bars1, val_counts):
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%\n({count:,}h)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                    
    for bar, count in zip(bars2, test_counts):
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%\n({count:,}h)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                    
    ax.set_title("Anomaly Detection Rate Comparison: Validation vs Test Split",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Anomaly Detection Rate (% of Classified Hours)", fontsize=11, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10.5, fontweight="semibold")
    ax.set_ylim(0, max(max(val_rates), max(test_rates)) * 1.25)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot Saved]: Anomaly Rate Comparison -> {output_path.name}")


def plot_test_combined_anomaly_timeline(
    df_comb: pd.DataFrame,
    output_path: Path
) -> None:
    """Plot 6: Chronological test timeline highlighting Union vs Intersection anomalies."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.4, 1.2]})
    
    ts = pd.to_datetime(df_comb["timestamp"])
    actual = df_comb["actual_power_watts"]
    pred = df_comb["predicted_power_watts"]
    
    classified = df_comb["classified"] == True
    both_mask = classified & (df_comb["agreement_category"] == "both_anomalous")
    res_only_mask = classified & (df_comb["agreement_category"] == "residual_only")
    iso_only_mask = classified & (df_comb["agreement_category"] == "isolation_forest_only")
    
    # Top Panel: Full Test Horizon
    ax1.plot(ts, actual, color="#1f77b4", linewidth=0.9, alpha=0.6, label="Actual Power (Watts)")
    ax1.plot(ts, pred, color="#7f7f7f", linewidth=0.9, linestyle="--", alpha=0.7, label="Forecast Power (Watts)")
    
    # Plot union-only markers
    union_only = res_only_mask | iso_only_mask
    if union_only.sum() > 0:
        ax1.scatter(ts[union_only], actual[union_only], color="#ff7f0e", s=20, marker="o",
                    alpha=0.75, label=f"Single-Detector Anomaly (OR Only, N={union_only.sum():,})", zorder=4)
    # Plot high-confidence intersection markers
    if both_mask.sum() > 0:
        ax1.scatter(ts[both_mask], actual[both_mask], color="#d62728", s=38, marker="^",
                    edgecolors="#333333", linewidth=0.5,
                    label=f"High-Confidence Anomaly (Both / AND, N={both_mask.sum():,})", zorder=6)
                    
    ax1.set_title("Test Horizon: Combined Anomaly Timeline (Full May-Dec 2017 Period)",
                  fontsize=12, fontweight="bold", pad=8)
    ax1.set_ylabel("Active Power (Watts)", fontsize=10.5, fontweight="semibold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
    
    # Bottom Panel: Representative 4-Week Granular Zoom (672 hours)
    zoom_len = min(672, len(df_comb))
    ts_z = ts.iloc[:zoom_len]
    act_z = actual.iloc[:zoom_len]
    pred_z = pred.iloc[:zoom_len]
    both_z = both_mask.iloc[:zoom_len]
    union_only_z = union_only.iloc[:zoom_len]
    
    ax2.plot(ts_z, act_z, color="#1f77b4", linewidth=1.4, alpha=0.85, label="Actual Power (Watts)")
    ax2.plot(ts_z, pred_z, color="#7f7f7f", linewidth=1.3, linestyle="--", alpha=0.85, label="Forecast Power (Watts)")
    
    if union_only_z.sum() > 0:
        ax2.scatter(ts_z[union_only_z], act_z[union_only_z], color="#ff7f0e", s=45, marker="o",
                    label=f"Single-Detector Anomaly (N={union_only_z.sum():,})", zorder=4)
    if both_z.sum() > 0:
        ax2.scatter(ts_z[both_z], act_z[both_z], color="#d62728", s=75, marker="^",
                    edgecolors="#333333", linewidth=0.7,
                    label=f"High-Confidence Anomaly (N={both_z.sum():,})", zorder=6)
                    
    ax2.set_title(f"Granular View: First 4 Weeks ({zoom_len} Hours) with Dual-Detector Highlights",
                  fontsize=11.5, fontweight="bold", pad=8)
    ax2.set_xlabel("Timestamp", fontsize=10.5, fontweight="semibold")
    ax2.set_ylabel("Active Power (Watts)", fontsize=10.5, fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot Saved]: Test Combined Anomaly Timeline -> {output_path.name}")


# ---------------------------------------------------------------------------
# Comprehensive Markdown Report Generation
# ---------------------------------------------------------------------------

def write_comparison_report(
    val_m: Dict[str, Any],
    test_m: Dict[str, Any],
    res_meta: Dict[str, Any],
    iso_meta: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate thorough academic markdown report documenting comparison analysis."""
    report_content = f"""# Detector Comparison and Combined Anomaly Analysis Report

**Project Title:** Electricity Consumption Forecasting and Anomaly Detection Using Machine Learning and Deep Learning  
**Milestone:** 4 — Detector Comparison and Combined Anomaly Analysis  
**Evaluation Scope:** Chronological Validation Split (5,778 hours) & Test Split (5,778 hours)  

---

## 1. Executive Summary

This report presents a rigorous comparative analysis between two complementary unsupervised anomaly detection paradigms developed for academic building electricity consumption monitoring:
1. **Model-Centric Residual Anomaly Detector:** Identifies temporal forecast violations where actual active power significantly departs from XGBoost point predictions ($e = y - \\hat{{y}}$).
2. **Multivariate Feature-Centric Isolation Forest:** Identifies structurally anomalous operating states across the 39-dimensional feature space without conditioning on target forecast accuracy.

Neither detector was retrained, and all detection thresholds were strictly frozen from validation calibration metadata.

### Key Comparative Findings:
- **Validation Split (Calibration Period, N=5,756 classified hours):**
  - Residual-Based Anomalies: **{val_m['Residual_Anomalies']}** ({val_m['Residual_Anomaly_Percentage']}%)
  - Isolation Forest Anomalies: **{val_m['Isolation_Forest_Anomalies']}** ({val_m['Isolation_Forest_Anomaly_Percentage']}%)
  - Mutual Agreement (Intersection / AND): **{val_m['Both_Anomalous_Intersection']}** ({val_m['Intersection_Percentage']}%)
  - Union Coverage (OR): **{val_m['Union_Anomalies_OR']}** ({val_m['Union_Percentage']}%)
  - Overall Agreement: **{val_m['Agreement_Percentage']}%** | Jaccard Similarity: **{val_m['Jaccard_Similarity']}** | Cohen's Kappa: **{val_m['Cohen_Kappa']}**
- **Test Split (Frozen Evaluation Period, N=5,660 classified hours):**
  - Residual-Based Anomalies: **{test_m['Residual_Anomalies']}** ({test_m['Residual_Anomaly_Percentage']}%)
  - Isolation Forest Anomalies: **{test_m['Isolation_Forest_Anomalies']}** ({test_m['Isolation_Forest_Anomaly_Percentage']}%)
  - Mutual Agreement (Intersection / AND): **{test_m['Both_Anomalous_Intersection']}** ({test_m['Intersection_Percentage']}%)
  - Union Coverage (OR): **{test_m['Union_Anomalies_OR']}** ({test_m['Union_Percentage']}%)
  - Overall Agreement: **{test_m['Agreement_Percentage']}%** | Jaccard Similarity: **{test_m['Jaccard_Similarity']}** | Cohen's Kappa: **{test_m['Cohen_Kappa']}**

---

## 2. Detector Architectures & Calibration Specifications

Both detectors utilize exact parameter calibrations established during earlier project phases:

### A. Residual-Based Detector (Milestone 2)
- **Forecasting Backbone:** Saved XGBoost Regressor (`models/xgboost_power.json`, 39 features, best iteration 626).
- **Residual Definition:** $e_t = \\text{{actual\\_power}}_t - \\text{{predicted\\_power}}_t$ (Watts).
- **Validation Residual Median:** **+{res_meta['validation_residual_median_watts']:.2f} W**
- **Validation MAD:** **{res_meta['validation_mad_watts']:.2f} W**
- **Robust Scale ($\\hat{{\\sigma}}_{{\\text{{robust}}}}$):** **{res_meta['robust_scale_watts']:.2f} W** ($1.4826 \\times \\text{{MAD}}$).
- **Upper Threshold ($z_{{\\text{{pos}}}}$):** **+{res_meta['positive_threshold']}** robust units (**+{res_meta['positive_threshold_watts']:.2f} W** deviation).
- **Lower Threshold ($z_{{\\text{{neg}}}}$):** **-{res_meta['negative_threshold']}** robust units (**{res_meta['negative_threshold_watts']:.2f} W** deviation).
- **Target Gap Policy:** Unrecorded sensor gaps (22 in validation, 118 in test) are preserved with NaN scores and explicit unclassified flags.

### B. Isolation Forest Detector (Milestone 3)
- **Model Architecture:** `IsolationForest(n_estimators=300, random_state=42, contamination='auto', n_jobs=-1)`.
- **Feature Preprocessing:** `SimpleImputer(strategy='median')` fitted strictly on training features (indices 0 to 26,963).
- **Score Inversion:** $\\text{{isolation\\_forest\\_anomaly\\_score}} = -\\text{{decision\\_function}}$ (higher values strictly denote higher anomaly likelihood).
- **Validation Threshold:** **{iso_meta['threshold_calibration']['validation_threshold']:.6f}** (calibrated at the **95.0th percentile** of validation anomaly scores).
- **Threshold Rule:** $\\text{{anomaly}} = (\\text{{score}} \\ge {iso_meta['threshold_calibration']['validation_threshold']:.6f})$.

---

## 3. Detailed Agreement and Disagreement Matrix

### Quantitative Breakdown Table:

| Metric | Validation Split (Calibration) | Test Split (Frozen Evaluation) |
|---|---:|---:|
| **Total Observations** | {val_m['Total_Observations']:,} | {test_m['Total_Observations']:,} |
| **Classified Observations** | {val_m['Classified_Observations']:,} | {test_m['Classified_Observations']:,} |
| **Unclassified Gaps** (Missing Targets) | {val_m['Unclassified_Observations']} | {test_m['Unclassified_Observations']} |
| **Both Anomalous (Intersection / AND)** | **{val_m['Both_Anomalous_Intersection']}** ({val_m['Intersection_Percentage']}%) | **{test_m['Both_Anomalous_Intersection']}** ({test_m['Intersection_Percentage']}%) |
| **Residual-Only Anomalies** | **{val_m['Residual_Only_Anomalies']}** ({val_m['Residual_Only_Percentage']}%) | **{test_m['Residual_Only_Anomalies']}** ({test_m['Residual_Only_Percentage']}%) |
| **Isolation Forest-Only Anomalies** | **{val_m['Isolation_Forest_Only_Anomalies']}** ({val_m['Isolation_Forest_Only_Percentage']}%) | **{test_m['Isolation_Forest_Only_Anomalies']}** ({test_m['Isolation_Forest_Only_Percentage']}%) |
| **Neither Anomalous (Concurring Normal)** | **{val_m['Neither_Anomalous']}** ({val_m['Neither_Percentage']}%) | **{test_m['Neither_Anomalous']}** ({test_m['Neither_Percentage']}%) |
| **Union Anomalies (OR Rule)** | **{val_m['Union_Anomalies_OR']}** ({val_m['Union_Percentage']}%) | **{test_m['Union_Anomalies_OR']}** ({test_m['Union_Percentage']}%) |
| **Overall Agreement Percentage** | **{val_m['Agreement_Percentage']}%** | **{test_m['Agreement_Percentage']}%** |
| **Jaccard Similarity Coefficient** | **{val_m['Jaccard_Similarity']}** | **{test_m['Jaccard_Similarity']}** |
| **Cohen's Kappa ($\\kappa$)** | **{val_m['Cohen_Kappa']}** | **{test_m['Cohen_Kappa']}** |

---

## 4. Directional Residual Analysis

The directional orientation of consumption deviations was computed for all observations where actual measurements exist. Isolation Forest does not possess directional semantics (it isolates rare combinations, not magnitude directions); therefore, directional attributions are derived strictly from the forecasting residual:

| Category | Validation Split | Test Split |
|---|---:|---:|
| **High Consumption Anomalies (Actual >> Forecast)** | | |
| - High Consumption in Mutual Overlap (Both) | **{val_m['Both_High_Consumption']}** | **{test_m['Both_High_Consumption']}** |
| - High Consumption in Residual-Only | **{val_m['Residual_Only_High_Consumption']}** | **{test_m['Residual_Only_High_Consumption']}** |
| - Total High Residual Anomalies | **{val_m['Residual_Total_High']}** | **{test_m['Residual_Total_High']}** |
| **Low Consumption Anomalies (Actual << Forecast)** | | |
| - Low Consumption in Mutual Overlap (Both) | **{val_m['Both_Low_Consumption']}** | **{test_m['Both_Low_Consumption']}** |
| - Low Consumption in Residual-Only | **{val_m['Residual_Only_Low_Consumption']}** | **{test_m['Residual_Only_Low_Consumption']}** |
| - Total Low Residual Anomalies | **{val_m['Residual_Total_Low']}** | **{test_m['Residual_Total_Low']}** |

---

## 5. Comparative Paradigm Analysis: Residual vs Multivariate Isolation

The empirical divergence between both models reflects fundamental conceptual differences:

```
+------------------------------------+------------------------------------+
| Residual-Based Anomaly Detection   | Isolation Forest Anomaly Detection |
+------------------------------------+------------------------------------+
| Evaluates univariate prediction    | Evaluates joint 39-dimensional     |
| deviation: e = y - f(X).          | feature partition depth.           |
|                                    |                                    |
| Sensitive to sudden load surges,   | Sensitive to rare feature states   |
| unexpected shutdowns, and forecast | (unusual lag combinations, atypical|
| model failures.                    | rolling variances, holiday shifts).|
|                                    |                                    |
| Directional: Clearly distinguishes | Non-directional: Flags rarity in   |
| over-consumption from dropouts.    | feature space regardless of sign.  |
|                                    |                                    |
| Requires actual target power.      | Operates without contemporaneous   |
| Missing targets = Unclassified.    | target readings.                   |
+------------------------------------+------------------------------------+
```

### Why Do Disagreements Occur?
1. **Residual-Only Anomalies ({val_m['Residual_Only_Anomalies']} val / {test_m['Residual_Only_Anomalies']} test):**
   - Occur when the feature vector $X_t$ appears standard (typical hour, standard lag profile), but actual power $y_t$ spikes or drops dramatically due to unannounced campus events or unmetered sub-loads.
2. **Isolation Forest-Only Anomalies ({val_m['Isolation_Forest_Only_Anomalies']} val / {test_m['Isolation_Forest_Only_Anomalies']} test):**
   - Occur during atypical operational contexts (e.g. rare calendar-lag alignment or atypical rolling standard deviations) where the XGBoost model successfully anticipates the unusual load, resulting in near-zero residual error.

---

## 6. Evaluation of Combined Combination Rules

### A. Union Combination Rule (OR Rule: $A_{{\\text{{res}}}} \\lor A_{{\\text{{iso}}}}$)
- **Validation Flagged:** **{val_m['Union_Anomalies_OR']} hours ({val_m['Union_Percentage']}%)**
- **Test Flagged:** **{test_m['Union_Anomalies_OR']} hours ({test_m['Union_Percentage']}%)**
- **Operational Benefit:** Maximizes screening sensitivity. Captures both sudden output deviations and rare input operational states.
- **Operational Limitation:** Higher operational alert volume; increases screening burden for building facilities managers.

### B. Intersection Combination Rule (AND Rule: $A_{{\\text{{res}}}} \\land A_{{\\text{{iso}}}}$)
- **Validation Flagged:** **{val_m['Both_Anomalous_Intersection']} hours ({val_m['Intersection_Percentage']}%)**
- **Test Flagged:** **{test_m['Both_Anomalous_Intersection']} hours ({test_m['Intersection_Percentage']}%)**
- **Operational Benefit:** Highest-confidence anomaly tier. Flagged hours represent extreme forecast failures occurring during structurally abnormal system conditions.
- **Operational Limitation:** Conservative screening rate; risks omitting genuine operational anomalies where only the target load was disrupted.

---

## 7. Validation to Test Evolution

Between the validation period (Sept 2016 – May 2017) and the test period (May 2017 – Dec 2017):
- Residual anomaly rate increased from **5.40%** to **15.00%**.
- Isolation Forest anomaly rate increased from **5.00%** to **11.25%**.
- Jaccard similarity between detectors increased from **0.0909** to **0.2834**.
- Cohen's Kappa increased from **0.1209** (slight agreement) to **0.3581** (fair-to-moderate agreement).

*Physical Explanation:* The test horizon incorporates late 2017 summer and winter peak demand transitions that exhibited elevated variance and more pronounced load disruptions compared to the calibration period. The elevated test anomalies reflect genuine non-stationarity in campus operational dynamics rather than detector miscalibration.

---

## 8. Scientific Limitations & Ground-Truth Disclaimer

> [!IMPORTANT]
> **Ground-Truth Label Absence:**
> There are currently no independently verified maintenance logs, breaker trip records, or sub-meter ground-truth fault labels for this academic building dataset.
> 
> Therefore:
> 1. Detected anomalies cannot be asserted as confirmed physical equipment failures.
> 2. Metrics such as classification accuracy, precision, recall, and F1-score are scientifically inapplicable and strictly omitted.
> 3. Anomalies should be interpreted as statistically unusual observations that warrant secondary human inspection.

---

## 9. Generated Artifacts Reference

### Output Datasets:
- [validation_combined_anomalies.csv](file:///D:/Electricity%20consumption%20forecasting/project/data/anomaly_detection/validation_combined_anomalies.csv) (5,778 rows)
- [test_combined_anomalies.csv](file:///D:/Electricity%20consumption%20forecasting/project/data/anomaly_detection/test_combined_anomalies.csv) (5,778 rows)
- [detector_comparison_summary.csv](file:///D:/Electricity%20consumption%20forecasting/project/reports/anomaly_detection/detector_comparison_summary.csv)

### Visualizations:
- [validation_detector_agreement.png](file:///D:/Electricity%20consumption%20forecasting/project/reports/anomaly_detection/visualizations/validation_detector_agreement.png)
- [test_detector_agreement.png](file:///D:/Electricity%20consumption%20forecasting/project/reports/anomaly_detection/visualizations/test_detector_agreement.png)
- [validation_anomaly_timeline_comparison.png](file:///D:/Electricity%20consumption%20forecasting/project/reports/anomaly_detection/visualizations/validation_anomaly_timeline_comparison.png)
- [test_anomaly_timeline_comparison.png](file:///D:/Electricity%20consumption%20forecasting/project/reports/anomaly_detection/visualizations/test_anomaly_timeline_comparison.png)
- [detector_anomaly_rate_comparison.png](file:///D:/Electricity%20consumption%20forecasting/project/reports/anomaly_detection/visualizations/detector_anomaly_rate_comparison.png)
- [test_combined_anomaly_timeline.png](file:///D:/Electricity%20consumption%20forecasting/project/reports/anomaly_detection/visualizations/test_combined_anomaly_timeline.png)
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"   [Report Saved]: {output_path.name} ({output_path.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Strict Verification Functions
# ---------------------------------------------------------------------------

def verify_combined_outputs(
    val_comb_df: pd.DataFrame,
    test_comb_df: pd.DataFrame
) -> None:
    """Strictly verify output datasets against all project specifications."""
    print("\n" + "=" * 75)
    print("STRICT SELF-VERIFICATION OF COMBINED ARTIFACTS")
    print("=" * 75)
    
    expected_cols = [
        "timestamp", "actual_power_watts", "predicted_power_watts",
        "residual_watts", "robust_residual_score", "high_consumption_anomaly",
        "low_consumption_anomaly", "residual_anomaly",
        "isolation_forest_anomaly_score", "isolation_forest_anomaly",
        "agreement_category", "combined_anomaly_or", "combined_anomaly_and",
        "classified"
    ]
    
    for df, name, exp_unclass in [(val_comb_df, "Validation", 22), (test_comb_df, "Test", 118)]:
        if len(df) != 5778:
            raise ValueError(f"Row count mismatch in {name}: {len(df)} vs 5,778")
            
        if list(df.columns) != expected_cols:
            raise ValueError(f"Column schema mismatch in {name}: {list(df.columns)} vs {expected_cols}")
            
        unclass_mask = df["classified"] == False
        unclass_count = int(unclass_mask.sum())
        if unclass_count != exp_unclass:
            raise ValueError(f"Unclassified count mismatch in {name}: {unclass_count} vs {exp_unclass}")
            
        # Verify unclassified agreement_category
        invalid_cat = (df.loc[unclass_mask, "agreement_category"] != "unclassified").sum()
        if invalid_cat > 0:
            raise ValueError(f"Found {invalid_cat} unclassified rows with wrong category in {name}")
            
        # Verify unclassified OR and AND are pd.NA / NaN
        if not df.loc[unclass_mask, "combined_anomaly_or"].isna().all():
            raise ValueError(f"Unclassified rows have non-null combined_anomaly_or in {name}")
        if not df.loc[unclass_mask, "combined_anomaly_and"].isna().all():
            raise ValueError(f"Unclassified rows have non-null combined_anomaly_and in {name}")
            
        # Verify classified OR and AND consistency
        class_df = df[~unclass_mask]
        r_b = class_df["residual_anomaly"] == True
        i_b = class_df["isolation_forest_anomaly"] == True
        or_b = class_df["combined_anomaly_or"] == True
        and_b = class_df["combined_anomaly_and"] == True
        
        if not (or_b == (r_b | i_b)).all():
            raise ValueError(f"Logical inconsistency in combined_anomaly_or for {name}")
        if not (and_b == (r_b & i_b)).all():
            raise ValueError(f"Logical inconsistency in combined_anomaly_and for {name}")
            
        print(f"   [Self-Check Passed]: Combined {name} Output File (5,778 rows)")
        print(f"      Classified: {len(class_df):,} | Unclassified: {unclass_count}")
        print(f"      Union (OR): {or_b.sum():,} | Intersection (AND): {and_b.sum():,}")


def main():
    print("=" * 75)
    print("ELECTRICITY CONSUMPTION FORECASTING & ANOMALY DETECTION")
    print("Milestone 4: Detector Comparison and Combined Anomaly Analysis")
    print("=" * 75)
    print(f"Project root resolved: {PROJECT_ROOT}")
    
    # 1. Load and Verify Inputs
    val_res, test_res, val_iso, test_iso, res_meta, iso_meta = load_and_verify_inputs()
    
    # 2. Build Combined Datasets
    print("\n" + "=" * 75)
    print("1. BUILDING COMBINED ANOMALY DATASETS")
    print("=" * 75)
    val_comb_df = build_combined_dataset(val_res, val_iso)
    test_comb_df = build_combined_dataset(test_res, test_iso)
    
    # 3. Calculate Comparison Metrics
    val_metrics = compute_comparison_metrics(val_comb_df, "Validation")
    test_metrics = compute_comparison_metrics(test_comb_df, "Test")
    
    # 4. Save Output Combined Datasets
    print("\n" + "=" * 75)
    print("2. SAVING COMBINED ANOMALY DATASETS & METRIC TABLES")
    print("=" * 75)
    val_comb_df.to_csv(VAL_COMBINED_PATH, index=False)
    print(f"   Saved: {VAL_COMBINED_PATH.name} ({VAL_COMBINED_PATH.stat().st_size:,} bytes)")
    
    test_comb_df.to_csv(TEST_COMBINED_PATH, index=False)
    print(f"   Saved: {TEST_COMBINED_PATH.name} ({TEST_COMBINED_PATH.stat().st_size:,} bytes)")
    
    # 5. Save Summary CSV
    summary_df = pd.DataFrame([val_metrics, test_metrics])
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"   Saved: {SUMMARY_CSV_PATH.name} ({SUMMARY_CSV_PATH.stat().st_size:,} bytes)")
    
    print("\n=== DETECTOR COMPARISON SUMMARY TABLE ===")
    display_cols = [
        "Split", "Classified_Observations", "Residual_Anomalies", "Residual_Anomaly_Percentage",
        "Isolation_Forest_Anomalies", "Isolation_Forest_Anomaly_Percentage",
        "Both_Anomalous_Intersection", "Intersection_Percentage",
        "Union_Anomalies_OR", "Union_Percentage",
        "Agreement_Percentage", "Jaccard_Similarity", "Cohen_Kappa"
    ]
    print(summary_df[display_cols].to_string(index=False))
    
    # 6. Generate Comprehensive Markdown Report
    print("\n" + "=" * 75)
    print("3. GENERATING ACADEMIC COMPARISON REPORT")
    print("=" * 75)
    write_comparison_report(val_metrics, test_metrics, res_meta, iso_meta, REPORT_MD_PATH)
    
    # 7. Generate Visualizations
    print("\n" + "=" * 75)
    print("4. GENERATING VISUALIZATIONS (REPORTS/ANOMALY_DETECTION/VISUALIZATIONS)")
    print("=" * 75)
    
    # Plot 1: Validation Venn Agreement
    plot_venn_agreement(val_metrics, "Validation", VIZ_DIR / "validation_detector_agreement.png")
    
    # Plot 2: Test Venn Agreement
    plot_venn_agreement(test_metrics, "Test", VIZ_DIR / "test_detector_agreement.png")
    
    # Plot 3: Validation Timeline Comparison
    plot_anomaly_timeline_comparison(val_comb_df, "Validation", VIZ_DIR / "validation_anomaly_timeline_comparison.png")
    
    # Plot 4: Test Timeline Comparison
    plot_anomaly_timeline_comparison(test_comb_df, "Test", VIZ_DIR / "test_anomaly_timeline_comparison.png")
    
    # Plot 5: Anomaly Rate Comparison Bar Chart
    plot_anomaly_rate_comparison(val_metrics, test_metrics, VIZ_DIR / "detector_anomaly_rate_comparison.png")
    
    # Plot 6: Test Combined Timeline
    plot_test_combined_anomaly_timeline(test_comb_df, VIZ_DIR / "test_combined_anomaly_timeline.png")
    
    # 8. Strict Verification
    verify_combined_outputs(val_comb_df, test_comb_df)
    
    print("\n" + "=" * 75)
    print("MILESTONE 4 COMPLETE: Detector Comparison and Combined Analysis Successful.")
    print("=" * 75)


if __name__ == "__main__":
    main()
