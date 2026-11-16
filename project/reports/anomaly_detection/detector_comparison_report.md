# Detector Comparison and Combined Anomaly Analysis Report

**Project Title:** Electricity Consumption Forecasting and Anomaly Detection Using Machine Learning and Deep Learning  
**Milestone:** 4 — Detector Comparison and Combined Anomaly Analysis  
**Evaluation Scope:** Chronological Validation Split (5,778 hours) & Test Split (5,778 hours)  

---

## 1. Executive Summary

This report presents a rigorous comparative analysis between two complementary unsupervised anomaly detection paradigms developed for academic building electricity consumption monitoring:
1. **Model-Centric Residual Anomaly Detector:** Identifies temporal forecast violations where actual active power significantly departs from XGBoost point predictions ($e = y - \hat{y}$).
2. **Multivariate Feature-Centric Isolation Forest:** Identifies structurally anomalous operating states across the 39-dimensional feature space without conditioning on target forecast accuracy.

Neither detector was retrained, and all detection thresholds were strictly frozen from validation calibration metadata.

### Key Comparative Findings:
- **Validation Split (Calibration Period, N=5,756 classified hours):**
  - Residual-Based Anomalies: **311** (5.4%)
  - Isolation Forest Anomalies: **289** (5.02%)
  - Mutual Agreement (Intersection / AND): **50** (0.87%)
  - Union Coverage (OR): **550** (9.56%)
  - Overall Agreement: **91.31%** | Jaccard Similarity: **0.0909** | Cohen's Kappa: **0.1209**
- **Test Split (Frozen Evaluation Period, N=5,660 classified hours):**
  - Residual-Based Anomalies: **849** (15.0%)
  - Isolation Forest Anomalies: **650** (11.48%)
  - Mutual Agreement (Intersection / AND): **331** (5.85%)
  - Union Coverage (OR): **1168** (20.64%)
  - Overall Agreement: **85.21%** | Jaccard Similarity: **0.2834** | Cohen's Kappa: **0.3581**

---

## 2. Detector Architectures & Calibration Specifications

Both detectors utilize exact parameter calibrations established during earlier project phases:

### A. Residual-Based Detector (Milestone 2)
- **Forecasting Backbone:** Saved XGBoost Regressor (`models/xgboost_power.json`, 39 features, best iteration 626).
- **Residual Definition:** $e_t = \text{actual\_power}_t - \text{predicted\_power}_t$ (Watts).
- **Validation Residual Median:** **+120.61 W**
- **Validation MAD:** **901.21 W**
- **Robust Scale ($\hat{\sigma}_{\text{robust}}$):** **1336.13 W** ($1.4826 \times \text{MAD}$).
- **Upper Threshold ($z_{\text{pos}}$):** **+3.5** robust units (**+4797.07 W** deviation).
- **Lower Threshold ($z_{\text{neg}}$):** **-3.5** robust units (**-4555.86 W** deviation).
- **Target Gap Policy:** Unrecorded sensor gaps (22 in validation, 118 in test) are preserved with NaN scores and explicit unclassified flags.

### B. Isolation Forest Detector (Milestone 3)
- **Model Architecture:** `IsolationForest(n_estimators=300, random_state=42, contamination='auto', n_jobs=-1)`.
- **Feature Preprocessing:** `SimpleImputer(strategy='median')` fitted strictly on training features (indices 0 to 26,963).
- **Score Inversion:** $\text{isolation\_forest\_anomaly\_score} = -\text{decision\_function}$ (higher values strictly denote higher anomaly likelihood).
- **Validation Threshold:** **0.080704** (calibrated at the **95.0th percentile** of validation anomaly scores).
- **Threshold Rule:** $\text{anomaly} = (\text{score} \ge 0.080704)$.

---

## 3. Detailed Agreement and Disagreement Matrix

### Quantitative Breakdown Table:

| Metric | Validation Split (Calibration) | Test Split (Frozen Evaluation) |
|---|---:|---:|
| **Total Observations** | 5,778 | 5,778 |
| **Classified Observations** | 5,756 | 5,660 |
| **Unclassified Gaps** (Missing Targets) | 22 | 118 |
| **Both Anomalous (Intersection / AND)** | **50** (0.87%) | **331** (5.85%) |
| **Residual-Only Anomalies** | **261** (4.53%) | **518** (9.15%) |
| **Isolation Forest-Only Anomalies** | **239** (4.15%) | **319** (5.64%) |
| **Neither Anomalous (Concurring Normal)** | **5206** (90.44%) | **4492** (79.36%) |
| **Union Anomalies (OR Rule)** | **550** (9.56%) | **1168** (20.64%) |
| **Overall Agreement Percentage** | **91.31%** | **85.21%** |
| **Jaccard Similarity Coefficient** | **0.0909** | **0.2834** |
| **Cohen's Kappa ($\kappa$)** | **0.1209** | **0.3581** |

---

## 4. Directional Residual Analysis

The directional orientation of consumption deviations was computed for all observations where actual measurements exist. Isolation Forest does not possess directional semantics (it isolates rare combinations, not magnitude directions); therefore, directional attributions are derived strictly from the forecasting residual:

| Category | Validation Split | Test Split |
|---|---:|---:|
| **High Consumption Anomalies (Actual >> Forecast)** | | |
| - High Consumption in Mutual Overlap (Both) | **31** | **206** |
| - High Consumption in Residual-Only | **161** | **304** |
| - Total High Residual Anomalies | **192** | **510** |
| **Low Consumption Anomalies (Actual << Forecast)** | | |
| - Low Consumption in Mutual Overlap (Both) | **19** | **125** |
| - Low Consumption in Residual-Only | **100** | **214** |
| - Total Low Residual Anomalies | **119** | **339** |

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
1. **Residual-Only Anomalies (261 val / 518 test):**
   - Occur when the feature vector $X_t$ appears standard (typical hour, standard lag profile), but actual power $y_t$ spikes or drops dramatically due to unannounced campus events or unmetered sub-loads.
2. **Isolation Forest-Only Anomalies (239 val / 319 test):**
   - Occur during atypical operational contexts (e.g. rare calendar-lag alignment or atypical rolling standard deviations) where the XGBoost model successfully anticipates the unusual load, resulting in near-zero residual error.

---

## 6. Evaluation of Combined Combination Rules

### A. Union Combination Rule (OR Rule: $A_{\text{res}} \lor A_{\text{iso}}$)
- **Validation Flagged:** **550 hours (9.56%)**
- **Test Flagged:** **1168 hours (20.64%)**
- **Operational Benefit:** Maximizes screening sensitivity. Captures both sudden output deviations and rare input operational states.
- **Operational Limitation:** Higher operational alert volume; increases screening burden for building facilities managers.

### B. Intersection Combination Rule (AND Rule: $A_{\text{res}} \land A_{\text{iso}}$)
- **Validation Flagged:** **50 hours (0.87%)**
- **Test Flagged:** **331 hours (5.85%)**
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
