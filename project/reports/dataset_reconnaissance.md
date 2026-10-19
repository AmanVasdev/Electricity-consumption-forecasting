# Dataset Reconnaissance Report: Academic Building Mains (`acad_build_mains.csv`)

**Date of Reconnaissance:** September 29, 2026  
**Target Dataset:** `energy_dataset/energy_dataset/acad_build_mains.csv`  
**Execution Environment:** `D:\Electricity consumption forecasting\.venv\Scripts\python.exe` (Pandas 3.0.6)  
**Sample File Generated:** `project/data/processed/acad_sample.csv` (10,000 rows)  

---

## 1. File Metadata & Overview

- **File Path:** `energy_dataset/energy_dataset/acad_build_mains.csv`
- **File Size:** `131,244,504 bytes` (~`125.16 MB`)
- **Total Records (Rows):** `2,269,530`
- **Reading Strategy:** Chunked reading (100,000 rows/chunk) to prevent memory overload.

---

## 2. Column Names & Inferred Data Types

| Column Name | Inferred Dtype | Storage Type | Description / Interpretation |
| :--- | :--- | :--- | :--- |
| `timestamp` | `int64` | Integer | UNIX epoch time in seconds |
| `power` | `float64` | Float | Active electrical power (in Watts) |
| `current` | `float64` | Float | Electrical current (in Amperes) |
| `voltage` | `float64` | Float | Supply voltage (in Volts) |
| `frequency` | `float64` | Float | Grid frequency (in Hertz) |
| `power_factor` | `float64` | Float | Electrical power factor (-1.0 to 1.0) |

---

## 3. First 5 Rows of Raw Data

| Index | timestamp | power (W) | current (A) | voltage (V) | frequency (Hz) | power_factor |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 1376073000 | 28368.60645 | `NaN` | 240.99646 | 50.32322 | 0.93397 |
| **1** | 1376073060 | 28411.98828 | `NaN` | 241.19509 | 50.35931 | 0.93536 |
| **2** | 1376073120 | 28752.53809 | `NaN` | 241.38574 | 50.39784 | 0.93675 |
| **3** | 1376073180 | 29031.44629 | `NaN` | 241.72626 | 50.44616 | 0.93685 |
| **4** | 1376073240 | 28487.45801 | `NaN` | 241.94244 | 50.44468 | 0.93448 |

---

## 4. Row Counts & Chunked Processing

- **Total Rows Measured:** `2,269,530`
- **Processing Approach:** Iterative chunk evaluation with 23 chunks of size 100,000 rows.
- **Verification:** Completed synchronously without memory spike.

---

## 5. Timestamp Analysis

### 5.1 Timestamp Format & Column Identification
- **Timestamp Column:** `timestamp`
- **Format:** UNIX epoch integer in seconds.
- **Target Timezone:** `Asia/Kolkata` (UTC+05:30) as per official I-BLEND dataset specification.

### 5.2 Timestamp Range & Timespan
- **Minimum Timestamp:** `1376073000`
  - **UTC:** `2013-08-09 18:30:00 UTC`
  - **Local (Asia/Kolkata):** `2013-08-10 00:00:00+05:30` (Midnight)
- **Maximum Timestamp:** `1514744940`
  - **UTC:** `2017-12-31 18:29:00 UTC`
  - **Local (Asia/Kolkata):** `2017-12-31 23:59:00+05:30`
- **Total Span:** `1,604 days, 23 hours, 59 minutes` (~`4.4 years`)
- **Total Theoretical 1-Minute Periods:** `2,311,200`
- **Actual Recorded Timestamps:** `2,269,530`
- **Missing Temporal Periods:** `41,670 minutes` (~`1.80%` of timeline)

### 5.3 Sampling Frequency Distribution

| Delta Between Timestamps | Duration | Occurrences | Percentage of Total Transitions | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **60 seconds** | 1.00 minute | **2,269,040** | **99.978%** | **Primary Sampling Frequency** |
| **120 seconds** | 2.00 minutes | 244 | 0.011% | 1 missed reading |
| **180 seconds** | 3.00 minutes | 91 | 0.004% | 2 missed readings |
| **240 seconds** | 4.00 minutes | 30 | 0.001% | 3 missed readings |
| **360 seconds** | 6.00 minutes | 29 | 0.001% | 5 missed readings |
| **300 seconds** | 5.00 minutes | 20 | 0.001% | 4 missed readings |
| **420 seconds** | 7.00 minutes | 10 | 0.000% | 6 missed readings |
| **480 seconds** | 8.00 minutes | 4 | 0.000% | 7 missed readings |
| **720 seconds** | 12.00 minutes | 3 | 0.000% | 11 missed readings |
| **900 seconds** | 15.00 minutes | 2 | 0.000% | 14 missed readings |

---

## 6. Duplicates & Sequence Integrity

- **Duplicate Timestamps:** `0` (100% unique timestamps across all 2,269,530 rows)
- **Non-Monotonic Steps:** `0` (Timestamps are strictly increasing without chronological reversals)
- **Exact Duplicate Rows:** `0`

---

## 7. Missing Values Summary

| Column | Missing Rows (Count) | Missing Percentage (%) | Observations |
| :--- | :--- | :--- | :--- |
| `timestamp` | 0 | 0.000% | Fully complete |
| `power` | 3 | 0.0001% | Virtually complete |
| `current` | 210,818 | 9.289% | Entirely missing from 2013-08-10 to 2014-01-04 (see section 10) |
| `voltage` | 3,836 | 0.169% | Minor sporadic sensor dropouts |
| `frequency` | 3 | 0.0001% | Virtually complete |
| `power_factor` | 2 | 0.0001% | Virtually complete |

---

## 8. Basic Statistics of Numerical Columns

Calculated across the entire 2,269,530 rows (with systematic distribution quantiles):

| Metric | `power` (W) | `voltage` (V) | `frequency` (Hz) | `power_factor` | `current` (A) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Valid Count** | 2,269,527 | 2,265,694 | 2,269,527 | 2,269,528 | 2,058,712 |
| **Mean** | 27,785.82 W | 241.22 V | 49.99 Hz | 0.6833 | 41.30 A |
| **Std Dev** | 14,018.36 W | 3.21 V | 0.17 Hz | 0.6545 | 21.06 A |
| **Min** | 0.00 W | 0.00 V | 0.00 Hz | -0.9990 | 0.00 A |
| **1st Percentile** | 8,109.37 W | 235.05 V | 49.73 Hz | -0.9982 | 12.62 A |
| **25th Percentile** | 18,126.85 W | 238.69 V | 49.94 Hz | 0.9187 | 26.93 A |
| **Median (50%)** | 23,335.35 W | 241.34 V | 49.99 Hz | 0.9470 | 34.32 A |
| **75th Percentile** | 33,233.08 W | 243.74 V | 50.05 Hz | 0.9701 | 48.63 A |
| **95th Percentile** | 57,564.10 W | 245.95 V | 50.14 Hz | 0.9977 | 86.33 A |
| **99th Percentile** | 67,527.46 W | 247.56 V | 50.25 Hz | 1.0000 | 101.34 A |
| **Max** | 95,786.33 W | 269.17 V | 52.85 Hz | 1.0000 | 144.94 A |

---

## 9. Electricity Consumption / Power Column Identification

- **Identified Target Column:** `power`
- **Unit of Measurement:** Watts (`W`), or kilowatts (`kW`) when divided by 1,000.
- **Physical Rationale:** 
  - Mean load: `27.79 kW`
  - Median load: `23.34 kW`
  - Peak load: `95.79 kW`
  - Baseload / 1st percentile: `8.11 kW`
  - This profile is consistent with an institutional academic facility (baseload overnight, peak active operational hours during day).
  - Corroborated by dataset `Readme.txt` stating: *"The power column in different files is in terms of watts."*

---

## 10. Data Quality Findings & Observations

### 10.1 Systematic Sensor Absence in `current`
- The `current` column is missing (`NaN`) for the first **210,818 rows**.
- **First valid timestamp:** `2014-01-04 01:40:00+05:30`.
- **Preceding period:** `2013-08-10 00:00:00` to `2014-01-04 01:39:00` has zero current readings. 
- After `2014-01-04 01:40:00`, current is recorded consistently without systematic gaps.

### 10.2 Zero and Abnormal Readings
- **Zero Power (`power == 0`):** `6,623` records (`0.29%` of dataset). Indicates building-level blackouts, generator transitions, or sensor uncouplings.
- **Zero Voltage & Zero Frequency (`voltage == 0`, `frequency == 0`):** Exactly `5` records where voltage and grid frequency simultaneously drop to `0.0`, reflecting complete electrical supply dropouts.
- **Negative Power Factor (`power_factor < 0`):** `312,849` records (`13.78%`). Negative power factors in power quality logging signify leading power factor (capacitive loads) or reverse reactive flow.

### 10.3 Top Recording Gaps (Timeline Discontinuities)
Although 99.98% of recorded timestamps occur at precise 1-minute intervals, there are 41,670 unrecorded minutes across 4.4 years. The largest contiguous data outages are:

1. **254.73 hours (10.61 days):** `2014-05-20 22:34:00+05:30` to `2014-05-31 13:18:00+05:30`
2. **91.82 hours (3.83 days):** `2017-08-05 12:42:00+05:30` to `2017-08-09 08:31:00+05:30`
3. **75.48 hours (3.15 days):** `2014-02-21 12:02:00+05:30` to `2014-02-24 15:31:00+05:30`
4. **69.48 hours (2.90 days):** `2014-02-24 16:33:00+05:30` to `2014-02-27 14:02:00+05:30`
5. **48.15 hours (2.01 days):** `2014-05-31 13:18:00+05:30` to `2014-06-02 13:27:00+05:30`

---

## 11. Created Artifacts & Deliverables

1. **`project/reports/dataset_reconnaissance.md`**: Complete report detailing the dataset characteristics.
2. **`project/data/processed/acad_sample.csv`**: First 10,000 rows preserved for rapid inspection and unit testing.
3. **`project/src/reconnaissance.py`**: Reproducible chunked processing script used for this audit.
4. **`project/reports/recon_raw_summary.json`**: Machine-readable JSON summary of audit metrics.
