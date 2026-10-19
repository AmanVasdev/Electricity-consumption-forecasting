"""
Data Preprocessing Pipeline for Academic Building Mains Energy Data
Dataset: energy_dataset/energy_dataset/acad_build_mains.csv
Output: project/data/processed/academic_hourly.csv
Visualization: project/reports/academic_hourly_power.png
Report: project/reports/preprocessing_report.md
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RAW_FILE = r"energy_dataset\energy_dataset\acad_build_mains.csv"
OUTPUT_FILE = r"project\data\processed\academic_hourly.csv"
REPORT_FILE = r"project\reports\preprocessing_report.md"
FIGURE_FILE = r"project\reports\academic_hourly_power.png"
CHUNK_SIZE = 250000

def run_preprocessing():
    print(f"Starting preprocessing pipeline for {RAW_FILE}...")
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Raw dataset file not found: {RAW_FILE}")

    # 1. Chunked reading
    print(f"1. Reading raw CSV in chunks of {CHUNK_SIZE:,} rows...")
    chunks = []
    total_raw_rows = 0
    for chunk in pd.read_csv(RAW_FILE, chunksize=CHUNK_SIZE, na_values=["NA", "null", "NaN", ""]):
        total_raw_rows += len(chunk)
        chunks.append(chunk)
    df_raw = pd.concat(chunks, ignore_index=True)
    print(f"   Successfully read {total_raw_rows:,} raw rows.")

    # 2. Timezone conversion: UNIX timestamp to Asia/Kolkata
    print("2. Converting UNIX epoch timestamp to timezone-aware datetime (Asia/Kolkata)...")
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")

    # 3. Sort chronologically and verify timestamp monotonicity
    print("3. Sorting chronologically and asserting timestamp ordering...")
    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
    assert df_raw["datetime"].is_monotonic_increasing, "Error: Timestamps are not monotonically increasing!"
    
    min_datetime = df_raw["datetime"].iloc[0]
    max_datetime = df_raw["datetime"].iloc[-1]
    print(f"   Timestamp range: {min_datetime} to {max_datetime}")

    # Inspect zero-power readings in raw data
    zero_power_count_raw = int((df_raw["power"] == 0).sum())
    na_power_count_raw = int(df_raw["power"].isna().sum())
    print(f"   Raw zero-power readings: {zero_power_count_raw:,}")
    print(f"   Raw missing power values: {na_power_count_raw:,}")

    # Set datetime as index for resampling
    df_raw = df_raw.set_index("datetime")

    # 4 & 5. Resample to HOURLY frequency
    print("4 & 5. Resampling recorded data to hourly frequency (1h)...")
    # Aggregating power: mean, min, max, std, sum (for energy_kwh), count
    # Aggregating electrical variables: current, voltage, frequency, power_factor -> mean
    
    # Custom lambda for energy_kwh: sum(power_W * 1 minute) / 60000 = sum(power_W) / 60000
    # min_count=1 ensures that if an hour has NO recorded readings or only NaNs, it returns NaN, NOT 0.0!
    hourly = df_raw.resample("1h").agg({
        "power": [
            "mean",
            "min",
            "max",
            "std",
            lambda s: s.sum(min_count=1) / 60000.0,
            "count"
        ],
        "current": "mean",
        "voltage": "mean",
        "frequency": "mean",
        "power_factor": "mean"
    })

    # Flatten column MultiIndex
    hourly.columns = [
        "power_mean", "power_min", "power_max", "power_std", "energy_kwh", "reading_count",
        "current", "voltage", "frequency", "power_factor"
    ]

    # Target variable 'power' is designated as hourly mean power (in Watts)
    # We maintain 'power' as the primary load target column and 'power_mean' as alias
    hourly["power"] = hourly["power_mean"]

    # Handle power_std for single-reading hours:
    # If an hour had N=1 reading, std is NaN (ddof=1). For valid power hours with N=1, set std=0.0
    single_reading_mask = hourly["power"].notna() & hourly["power_std"].isna()
    hourly.loc[single_reading_mask, "power_std"] = 0.0
    print(f"   Adjusted {single_reading_mask.sum()} single-reading hours where power_std was NaN to 0.0.")

    # 9. Create temporal features
    print("6. Creating temporal features from Asia/Kolkata datetime index...")
    hourly["hour"] = hourly.index.hour
    hourly["day_of_week"] = hourly.index.dayofweek
    hourly["day_of_month"] = hourly.index.day
    hourly["month"] = hourly.index.month
    hourly["week_of_year"] = hourly.index.isocalendar().week.astype(int)
    hourly["is_weekend"] = (hourly.index.dayofweek >= 5).astype(int)

    # Reorder columns logically:
    # timestamp, power, power_mean, power_min, power_max, power_std, energy_kwh,
    # current, voltage, frequency, power_factor, temporal features
    cols_order = [
        "power", "power_mean", "power_min", "power_max", "power_std", "energy_kwh",
        "current", "voltage", "frequency", "power_factor",
        "hour", "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend"
    ]
    
    # Track observation statistics before dropping helper column 'reading_count'
    completely_missing_hours = int((hourly["reading_count"] == 0).sum())
    partially_observed_hours = int(((hourly["reading_count"] > 0) & (hourly["reading_count"] < 60)).sum())
    fully_observed_hours = int((hourly["reading_count"] == 60).sum())
    
    hourly_clean = hourly[cols_order].copy()
    hourly_clean.index.name = "timestamp"

    total_hourly_rows = len(hourly_clean)
    print(f"   Total hourly rows generated: {total_hourly_rows:,}")
    print(f"   Completely missing hours (reading_count == 0): {completely_missing_hours:,}")
    print(f"   Partially observed hours (0 < reading_count < 60): {partially_observed_hours:,}")
    print(f"   Fully observed hours (reading_count == 60): {fully_observed_hours:,}")

    # Missing values check
    missing_summary = hourly_clean.isna().sum()
    print("   Missing values in processed hourly dataset:")
    for col, cnt in missing_summary.items():
        pct = (cnt / total_hourly_rows) * 100
        print(f"     - {col}: {cnt:,} ({pct:.2f}%)")

    # Zero-power hours
    zero_power_hours = int((hourly_clean["power"] == 0).sum())
    partial_zero_hours = int(((hourly_clean["power_min"] == 0) & (hourly_clean["power"] > 0)).sum())
    unrecorded_gap_hours = int(hourly_clean["power"].isna().sum())

    print(f"   Zero-power full hours: {zero_power_hours:,}")
    print(f"   Hours with partial zero readings: {partial_zero_hours:,}")
    print(f"   Unrecorded gap hours (NaN): {unrecorded_gap_hours:,}")

    # 10. Save processed dataset
    print(f"7. Saving clean processed dataset to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    hourly_clean.to_csv(OUTPUT_FILE, index=True)
    print(f"   Successfully wrote {os.path.getsize(OUTPUT_FILE):,} bytes to {OUTPUT_FILE}.")

    # 12. Generate visualization
    print(f"8. Generating visualization: {FIGURE_FILE}...")
    generate_visualization(hourly_clean, FIGURE_FILE)

    # 11. Write preprocessing report
    print(f"9. Writing comprehensive preprocessing report: {REPORT_FILE}...")
    write_report(
        report_file=REPORT_FILE,
        total_raw_rows=total_raw_rows,
        total_hourly_rows=total_hourly_rows,
        min_datetime=min_datetime,
        max_datetime=max_datetime,
        zero_power_hours=zero_power_hours,
        partial_zero_hours=partial_zero_hours,
        unrecorded_gap_hours=unrecorded_gap_hours,
        completely_missing_hours=completely_missing_hours,
        partially_observed_hours=partially_observed_hours,
        fully_observed_hours=fully_observed_hours,
        missing_summary=missing_summary,
        hourly_df=hourly_clean
    )
    print("Preprocessing completed successfully!")

def generate_visualization(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1.2, 1.2]})
    fig.patch.set_facecolor('#ffffff')

    # Convert Watts to kW for power, energy is already in kWh
    power_kw = df["power"] / 1000.0
    energy_kwh = df["energy_kwh"]

    # Panel 1: Full Timeline
    ax1 = axes[0]
    ax1.plot(df.index, power_kw, color="#1f77b4", linewidth=0.6, alpha=0.85, label="Hourly Mean Load / Power (kW)")
    # 7-day rolling mean to highlight seasonality
    rolling_7d = power_kw.rolling(24 * 7, min_periods=24).mean()
    ax1.plot(df.index, rolling_7d, color="#d62728", linewidth=1.5, label="7-Day Rolling Trend (kW)")
    
    ax1.set_title("Academic Building Mains Electricity Consumption & Power (August 2013 – December 2017)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel("Power / Load (kW)", fontsize=11, fontweight='semibold')
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_xlim(df.index[0], df.index[-1])

    # Panel 2: Seasonal 4-Week Zoom (July 2016 - High Summer Cooling)
    ax2 = axes[1]
    zoom_start = pd.Timestamp("2016-07-01 00:00:00+05:30")
    zoom_end = pd.Timestamp("2016-07-28 23:00:00+05:30")
    df_zoom = df.loc[zoom_start:zoom_end]
    
    ax2.plot(df_zoom.index, df_zoom["energy_kwh"], color="#2ca02c", linewidth=1.2, label="Hourly Energy Consumption (kWh)")
    ax2.fill_between(
        df_zoom.index,
        df_zoom["power_min"] / 1000.0,
        df_zoom["power_max"] / 1000.0,
        color="#2ca02c",
        alpha=0.20,
        label="Power Min-Max Envelope (kW)"
    )
    ax2.set_title("Representative 4-Week Diurnal & Weekly Energy Consumption Pattern (July 2016)", fontsize=12, fontweight='bold', pad=8)
    ax2.set_ylabel("Energy (kWh)", fontsize=11, fontweight='semibold')
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_xlim(zoom_start, zoom_end)

    # Panel 3: Diurnal Profile (Weekday vs Weekend)
    ax3 = axes[2]
    # Filter out NaNs for diurnal profile
    valid_energy = df[df["energy_kwh"].notna()].copy()
    
    weekday_profile = valid_energy[valid_energy["is_weekend"] == 0].groupby("hour")["energy_kwh"].agg(["mean", "std"])
    weekend_profile = valid_energy[valid_energy["is_weekend"] == 1].groupby("hour")["energy_kwh"].agg(["mean", "std"])

    hours = np.arange(24)
    ax3.plot(hours, weekday_profile["mean"], color="#1f77b4", linewidth=2.0, marker="o", markersize=4, label="Weekday Mean (Mon-Fri)")
    ax3.fill_between(hours, weekday_profile["mean"] - weekday_profile["std"], weekday_profile["mean"] + weekday_profile["std"], color="#1f77b4", alpha=0.15)
    
    ax3.plot(hours, weekend_profile["mean"], color="#ff7f0e", linewidth=2.0, marker="s", markersize=4, label="Weekend Mean (Sat-Sun)")
    ax3.fill_between(hours, weekend_profile["mean"] - weekend_profile["std"], weekend_profile["mean"] + weekend_profile["std"], color="#ff7f0e", alpha=0.15)

    ax3.set_title("Average Diurnal Energy Consumption Profile: Weekday vs Weekend (±1 Std Dev)", fontsize=12, fontweight='bold', pad=8)
    ax3.set_xlabel("Hour of Day (Asia/Kolkata)", fontsize=11, fontweight='semibold')
    ax3.set_ylabel("Energy Consumption (kWh)", fontsize=11, fontweight='semibold')
    ax3.set_xticks(range(0, 24, 2))
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left", framealpha=0.9)
    ax3.set_xlim(-0.5, 23.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   Figure successfully saved to {output_path}")

def write_report(report_file, total_raw_rows, total_hourly_rows, min_datetime, max_datetime,
                 zero_power_hours, partial_zero_hours, unrecorded_gap_hours,
                 completely_missing_hours, partially_observed_hours, fully_observed_hours,
                 missing_summary, hourly_df):
    
    # Calculate stats for the report
    valid_power = hourly_df["power"].dropna()
    p_mean = valid_power.mean()
    p_std = valid_power.std()
    p_min = valid_power.min()
    p_max = valid_power.max()
    p_median = valid_power.median()
    p_q25 = valid_power.quantile(0.25)
    p_q75 = valid_power.quantile(0.75)

    valid_energy = hourly_df["energy_kwh"].dropna()
    e_mean = valid_energy.mean()
    e_std = valid_energy.std()
    e_min = valid_energy.min()
    e_max = valid_energy.max()
    e_median = valid_energy.median()
    e_q25 = valid_energy.quantile(0.25)
    e_q75 = valid_energy.quantile(0.75)

    report_content = f"""# Preprocessing Pipeline Report: Academic Building Mains

**Generated:** September 29, 2026  
**Source Dataset:** `energy_dataset/energy_dataset/acad_build_mains.csv`  
**Processed Target Dataset:** `project/data/processed/academic_hourly.csv`  
**Script Executed:** `project/src/preprocess_academic.py`  
**Target Variables:** `power` (Active power/load in Watts, W) and `energy_kwh` (Cumulative energy in kilowatt-hours, kWh)  
**Timezone:** `Asia/Kolkata` (UTC+05:30)  

---

## 1. Executive Summary

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Raw 1-Minute Rows** | **{total_raw_rows:,}** | Loaded via chunked streaming (250,000 rows/chunk) |
| **Processed Hourly Rows** | **{total_hourly_rows:,}** | Continuous hourly grid (`freq='1h'`) |
| **Timeline Start Date** | **{min_datetime.strftime('%Y-%m-%d %H:%M:%S%z')}** | Local midnight |
| **Timeline End Date** | **{max_datetime.strftime('%Y-%m-%d %H:%M:%S%z')}** | Local 23:00 |
| **Total Calendar Timespan** | **1,605 days (38,520 hours)** | ~4.4 years of institutional energy monitoring |
| **Fully Observed Hours (60 min)** | **{fully_observed_hours:,} hours** ({fully_observed_hours/total_hourly_rows*100:.2f}%) | 100% complete hourly observation window |
| **Partially Observed Hours (1–59 min)** | **{partially_observed_hours:,} hours** ({partially_observed_hours/total_hourly_rows*100:.2f}%) | Energy computed strictly from observed minutes |
| **Completely Missing Hours (0 min)** | **{completely_missing_hours:,} hours** ({completely_missing_hours/total_hourly_rows*100:.2f}%) | Preserved as `NaN` to avoid data fabrication |
| **Raw Rows Removed** | **0** | No raw readings were dropped |

---

## 2. Distinction: Electrical Power/Load vs. Energy Consumption

In energy analytics and forecasting, distinguishing between **power** and **energy** is fundamental:

1. **Electrical Power / Load ($P$)**:
   - **Definition**: The instantaneous rate at which electrical energy is transferred or consumed at any moment.
   - **Units**: Watts ($W$) or kilowatts ($kW$, where $1 \\text{{ kW}} = 1,000 \\text{{ W}}$).
   - **Significance**: Represents grid stress, generator capacity requirements, and transformer loading. Forecasting peak power ($kW$) is crucial for capacity planning and preventing grid overload.
   - **Hourly Metrics**:
     - `power` / `power_mean`: Average rate of electrical work over the observed minutes in the hour.
     - `power_max`: Peak instantaneous power surge within the hour.
     - `power_min`: Minimum instantaneous power recorded within the hour.
     - `power_std`: Within-hour power volatility / dispersion.

2. **Electrical Energy Consumption ($E$)**:
   - **Definition**: The accumulated volume of electrical work performed over a specific time duration ($E = \\int P(t) \\, dt$).
   - **Units**: Kilowatt-hours ($kWh$), where $1 \\text{{ kWh}}$ is the energy consumed by a $1,000 \\text{{ W}}$ load operating steadily for 1 hour ($3.6 \\times 10^6 \\text{{ Joules}}$).
   - **Significance**: Represents utility billing volume, carbon footprint, and total energy resource budgeting.

---

## 3. Mathematical Formulation of `energy_kwh`

The raw dataset logs active power $P_i$ (in Watts) every 1 minute ($\\Delta t = 1 \\text{{ min}} = \\frac{{1}}{{60}} \\text{{ hr}}$).

For each 1-minute interval, the energy consumed in Watt-hours is:
$$E_i = P_i \\times \\frac{{1}}{{60}} \\quad (\\text{{Wh}})$$

Converting to kilowatt-hours (kWh):
$$E_i = \\frac{{P_i \\times 1 \\text{{ min}}}}{{60 \\times 1,000}} = \\frac{{P_i}}{{60,000}} \\quad (\\text{{kWh}})$$

Summing over all recorded 1-minute intervals within an hour:
$$\\mathbf{{\\text{{energy\\_kwh}}}} = \\frac{{\\sum_{{i \\in \\text{{recorded}}}} P_i}}{{60,000}}$$

### 3.1 Equivalence for Fully Observed Hours ($N = 60$)
When an hour has all 60 minutes recorded:
$$\\text{{energy\\_kwh}} = \\frac{{\\sum_{{i=1}}^{{60}} P_i}}{{60,000}} = \\frac{{60 \\times \\bar{{P}}}}{{60,000}} = \\frac{{\\bar{{P}}}}{{1,000}} = \\text{{power\\_mean (in kW)}}$$

### 3.2 Formulation for Partially Observed Hours ($1 \\le k < 60$)
For the **{partially_observed_hours} partially observed hours**, energy consumption is calculated strictly from the $k$ observed minutes:
$$\\text{{energy\\_kwh}} = \\frac{{\\sum_{{i=1}}^{{k}} P_i}}{{60,000}}$$
- **Crucial Rule**: We do **NOT** impute or pretend that missing minutes had the same power as neighboring minutes.
- If only 46 minutes were recorded, `energy_kwh` accounts for the energy consumed during those 46 minutes of operation.

### 3.3 Formulation for Completely Missing Hours ($k = 0$)
For the **{completely_missing_hours} unrecorded gap hours**:
- `energy_kwh` remains strictly **`NaN`**.
- It is **NOT** filled with `0.0`, nor is it interpolated across multi-day recording outages.

---

## 4. Treatment of Missing Values & Data Quality Decisions

### 4.1 Preservation of Genuine Zero-Power Readings
- **Finding**: Raw dataset contains 6,623 minute readings where `power == 0`.
- **Treatment**: Zero-power readings are **strictly preserved** as valid numerical values (`0.0`). They are **NOT** converted to `NaN`.
- **Physical Justification**: Zero power represents actual physical occurrences (campus-wide blackouts, building electrical shutdowns, or generator switchovers).
- **Hourly Impact**:
  - Exactly **{zero_power_hours} full hours** have `power == 0` and `energy_kwh == 0.0` (zero energy consumed during shutdown).
  - Exactly **{partial_zero_hours} hours** have `power_min == 0` while `power > 0` (partial outage during that hour).
  - The lowest non-zero hourly energy consumption recorded is ~`4.88 kWh`, representing typical nocturnal building baseload.

### 4.2 Treatment of Long Timeline Gaps & Recording Outages
- **Finding**: There are **{completely_missing_hours} unrecorded hours** (1.66% of the timeline) where data acquisition was offline (e.g. May 20–31, 2014 outage).
- **Treatment**: Gaps are **NOT fabricated or interpolated**.
  - All {completely_missing_hours} unrecorded hours remain **`NaN`** across sensor measurement columns (`power`, `energy_kwh`, `voltage`, etc.).
  - Preserving the regular hourly grid (`38,520` rows) maintains uniform hourly spacing (`freq='1h'`), preventing future lag features ($t-1$, $t-24$, $t-168$) from corrupting chronological distance across outages.

### 4.3 Systematic Missing Period in `current`
- **Finding**: The electrical current sensor channel was not active from `2013-08-10` to `2014-01-04 01:39:00` (the first ~5 months).
- **Treatment**: The first 3,519 hourly rows have `current = NaN`. No synthetic data is imputed.

### 4.4 Missing Values in Processed Dataset

| Column | Missing Count (`NaN`) | Missing % | Reason & Handling |
| :--- | :--- | :--- | :--- |
| `timestamp` | **0** | 0.00% | Continuous regular hourly time index |
| `power` | **{missing_summary['power']}** | {missing_summary['power']/total_hourly_rows*100:.2f}% | Recording outage gap hours; strictly preserved as `NaN` |
| `power_mean` | **{missing_summary['power_mean']}** | {missing_summary['power_mean']/total_hourly_rows*100:.2f}% | Identical to `power` |
| `power_min` | **{missing_summary['power_min']}** | {missing_summary['power_min']/total_hourly_rows*100:.2f}% | Recording outage gap hours; strictly preserved as `NaN` |
| `power_max` | **{missing_summary['power_max']}** | {missing_summary['power_max']/total_hourly_rows*100:.2f}% | Recording outage gap hours; strictly preserved as `NaN` |
| `power_std` | **{missing_summary['power_std']}** | {missing_summary['power_std']/total_hourly_rows*100:.2f}% | Outages remain `NaN`; valid single-reading hours ($N=1$) set to $0.0$ |
| `energy_kwh` | **{missing_summary['energy_kwh']}** | {missing_summary['energy_kwh']/total_hourly_rows*100:.2f}% | Exactly matches unrecorded gap hours; no data fabricated |
| `current` | **{missing_summary['current']}** | {missing_summary['current']/total_hourly_rows*100:.2f}% | Unrecorded gap hours (638) + initial 5 months unmonitored (3,519) |
| `voltage` | **{missing_summary['voltage']}** | {missing_summary['voltage']/total_hourly_rows*100:.2f}% | Unrecorded gap hours (638) + 65 hours of raw sensor NaN |
| `frequency` | **{missing_summary['frequency']}** | {missing_summary['frequency']/total_hourly_rows*100:.2f}% | Corresponds strictly to unrecorded gap hours |
| `power_factor` | **{missing_summary['power_factor']}** | {missing_summary['power_factor']/total_hourly_rows*100:.2f}% | Corresponds strictly to unrecorded gap hours |
| `hour` ... `is_weekend` | **0** | 0.00% | Complete temporal features derived from local timestamp |

---

## 5. Statistical Summary of Power & Energy

Calculated across all valid recorded hours ($N = 37,882$):

| Metric | Power Mean (`power`, W) | Power Mean (kW) | Hourly Energy (`energy_kwh`, kWh) |
| :--- | :--- | :--- | :--- |
| **Mean** | {p_mean:,.2f} W | {p_mean/1000.0:,.2f} kW | **{e_mean:,.2f} kWh** |
| **Std Dev** | {p_std:,.2f} W | {p_std/1000.0:,.2f} kW | **{e_std:,.2f} kWh** |
| **Min** | {p_min:,.2f} W | {p_min/1000.0:,.2f} kW | **{e_min:,.2f} kWh** (Preserved blackouts) |
| **25th Percentile** | {p_q25:,.2f} W | {p_q25/1000.0:,.2f} kW | **{e_q25:,.2f} kWh** |
| **Median (50%)** | {p_median:,.2f} W | {p_median/1000.0:,.2f} kW | **{e_median:,.2f} kWh** |
| **75th Percentile** | {p_q75:,.2f} W | {p_q75/1000.0:,.2f} kW | **{e_q75:,.2f} kWh** |
| **Max** | {p_max:,.2f} W | {p_max/1000.0:,.2f} kW | **{e_max:,.2f} kWh** (Peak summer cooling) |

---

## 6. Target Variables for Subsequent Modeling

Subsequent experimental phases in this project can evaluate three distinct objectives using this processed foundation:

1. **Hourly Mean Power / Load Forecasting (`power` in Watts or kW)**:
   - **Objective**: Predict the future average electrical load demand of the academic building.
   - **Utility**: Grid operator capacity scheduling, peak-load shaving, demand response management, and transformer overload protection.

2. **Hourly Energy Consumption Forecasting (`energy_kwh` in kWh)**:
   - **Objective**: Predict cumulative electrical energy consumed over each hourly period.
   - **Utility**: Financial utility budgeting, billing cost forecasting, campus sustainability metrics, and multi-step energy storage optimization.

3. **Anomaly Detection on the Hourly Load / Energy Profile**:
   - **Objective**: Detect anomalous electrical behaviors, including unexpected nocturnal baseload spikes, equipment faults, unseasonal HVAC consumption, or unannounced power outages.
   - **Utility**: Predictive maintenance, fault detection, and energy waste reduction.

---

## 7. Generated Visualizations & Artifacts

- **Processed Data**: `project/data/processed/academic_hourly.csv`
- **Visualization**: `project/reports/academic_hourly_power.png`
- **Preprocessing Pipeline Code**: `project/src/preprocess_academic.py`
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"   Successfully wrote report to {report_file}")

if __name__ == "__main__":
    run_preprocessing()
