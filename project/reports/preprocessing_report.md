# Preprocessing Pipeline Report: Academic Building Mains

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
| **Raw 1-Minute Rows** | **2,269,530** | Loaded via chunked streaming (250,000 rows/chunk) |
| **Processed Hourly Rows** | **38,520** | Continuous hourly grid (`freq='1h'`) |
| **Timeline Start Date** | **2013-08-10 00:00:00+0530** | Local midnight |
| **Timeline End Date** | **2017-12-31 23:59:00+0530** | Local 23:00 |
| **Total Calendar Timespan** | **1,605 days (38,520 hours)** | ~4.4 years of institutional energy monitoring |
| **Fully Observed Hours (60 min)** | **37,638 hours** (97.71%) | 100% complete hourly observation window |
| **Partially Observed Hours (1–59 min)** | **244 hours** (0.63%) | Energy computed strictly from observed minutes |
| **Completely Missing Hours (0 min)** | **638 hours** (1.66%) | Preserved as `NaN` to avoid data fabrication |
| **Raw Rows Removed** | **0** | No raw readings were dropped |

---

## 2. Distinction: Electrical Power/Load vs. Energy Consumption

In energy analytics and forecasting, distinguishing between **power** and **energy** is fundamental:

1. **Electrical Power / Load ($P$)**:
   - **Definition**: The instantaneous rate at which electrical energy is transferred or consumed at any moment.
   - **Units**: Watts ($W$) or kilowatts ($kW$, where $1 \text{ kW} = 1,000 \text{ W}$).
   - **Significance**: Represents grid stress, generator capacity requirements, and transformer loading. Forecasting peak power ($kW$) is crucial for capacity planning and preventing grid overload.
   - **Hourly Metrics**:
     - `power` / `power_mean`: Average rate of electrical work over the observed minutes in the hour.
     - `power_max`: Peak instantaneous power surge within the hour.
     - `power_min`: Minimum instantaneous power recorded within the hour.
     - `power_std`: Within-hour power volatility / dispersion.

2. **Electrical Energy Consumption ($E$)**:
   - **Definition**: The accumulated volume of electrical work performed over a specific time duration ($E = \int P(t) \, dt$).
   - **Units**: Kilowatt-hours ($kWh$), where $1 \text{ kWh}$ is the energy consumed by a $1,000 \text{ W}$ load operating steadily for 1 hour ($3.6 \times 10^6 \text{ Joules}$).
   - **Significance**: Represents utility billing volume, carbon footprint, and total energy resource budgeting.

---

## 3. Mathematical Formulation of `energy_kwh`

The raw dataset logs active power $P_i$ (in Watts) every 1 minute ($\Delta t = 1 \text{ min} = \frac{1}{60} \text{ hr}$).

For each 1-minute interval, the energy consumed in Watt-hours is:
$$E_i = P_i \times \frac{1}{60} \quad (\text{Wh})$$

Converting to kilowatt-hours (kWh):
$$E_i = \frac{P_i \times 1 \text{ min}}{60 \times 1,000} = \frac{P_i}{60,000} \quad (\text{kWh})$$

Summing over all recorded 1-minute intervals within an hour:
$$\mathbf{\text{energy\_kwh}} = \frac{\sum_{i \in \text{recorded}} P_i}{60,000}$$

### 3.1 Equivalence for Fully Observed Hours ($N = 60$)
When an hour has all 60 minutes recorded:
$$\text{energy\_kwh} = \frac{\sum_{i=1}^{60} P_i}{60,000} = \frac{60 \times \bar{P}}{60,000} = \frac{\bar{P}}{1,000} = \text{power\_mean (in kW)}$$

### 3.2 Formulation for Partially Observed Hours ($1 \le k < 60$)
For the **244 partially observed hours**, energy consumption is calculated strictly from the $k$ observed minutes:
$$\text{energy\_kwh} = \frac{\sum_{i=1}^{k} P_i}{60,000}$$
- **Crucial Rule**: We do **NOT** impute or pretend that missing minutes had the same power as neighboring minutes.
- If only 46 minutes were recorded, `energy_kwh` accounts for the energy consumed during those 46 minutes of operation.

### 3.3 Formulation for Completely Missing Hours ($k = 0$)
For the **638 unrecorded gap hours**:
- `energy_kwh` remains strictly **`NaN`**.
- It is **NOT** filled with `0.0`, nor is it interpolated across multi-day recording outages.

---

## 4. Treatment of Missing Values & Data Quality Decisions

### 4.1 Preservation of Genuine Zero-Power Readings
- **Finding**: Raw dataset contains 6,623 minute readings where `power == 0`.
- **Treatment**: Zero-power readings are **strictly preserved** as valid numerical values (`0.0`). They are **NOT** converted to `NaN`.
- **Physical Justification**: Zero power represents actual physical occurrences (campus-wide blackouts, building electrical shutdowns, or generator switchovers).
- **Hourly Impact**:
  - Exactly **107 full hours** have `power == 0` and `energy_kwh == 0.0` (zero energy consumed during shutdown).
  - Exactly **26 hours** have `power_min == 0` while `power > 0` (partial outage during that hour).
  - The lowest non-zero hourly energy consumption recorded is ~`4.88 kWh`, representing typical nocturnal building baseload.

### 4.2 Treatment of Long Timeline Gaps & Recording Outages
- **Finding**: There are **638 unrecorded hours** (1.66% of the timeline) where data acquisition was offline (e.g. May 20–31, 2014 outage).
- **Treatment**: Gaps are **NOT fabricated or interpolated**.
  - All 638 unrecorded hours remain **`NaN`** across sensor measurement columns (`power`, `energy_kwh`, `voltage`, etc.).
  - Preserving the regular hourly grid (`38,520` rows) maintains uniform hourly spacing (`freq='1h'`), preventing future lag features ($t-1$, $t-24$, $t-168$) from corrupting chronological distance across outages.

### 4.3 Systematic Missing Period in `current`
- **Finding**: The electrical current sensor channel was not active from `2013-08-10` to `2014-01-04 01:39:00` (the first ~5 months).
- **Treatment**: The first 3,519 hourly rows have `current = NaN`. No synthetic data is imputed.

### 4.4 Missing Values in Processed Dataset

| Column | Missing Count (`NaN`) | Missing % | Reason & Handling |
| :--- | :--- | :--- | :--- |
| `timestamp` | **0** | 0.00% | Continuous regular hourly time index |
| `power` | **638** | 1.66% | Recording outage gap hours; strictly preserved as `NaN` |
| `power_mean` | **638** | 1.66% | Identical to `power` |
| `power_min` | **638** | 1.66% | Recording outage gap hours; strictly preserved as `NaN` |
| `power_max` | **638** | 1.66% | Recording outage gap hours; strictly preserved as `NaN` |
| `power_std` | **638** | 1.66% | Outages remain `NaN`; valid single-reading hours ($N=1$) set to $0.0$ |
| `energy_kwh` | **638** | 1.66% | Exactly matches unrecorded gap hours; no data fabricated |
| `current` | **4157** | 10.79% | Unrecorded gap hours (638) + initial 5 months unmonitored (3,519) |
| `voltage` | **703** | 1.83% | Unrecorded gap hours (638) + 65 hours of raw sensor NaN |
| `frequency` | **638** | 1.66% | Corresponds strictly to unrecorded gap hours |
| `power_factor` | **638** | 1.66% | Corresponds strictly to unrecorded gap hours |
| `hour` ... `is_weekend` | **0** | 0.00% | Complete temporal features derived from local timestamp |

---

## 5. Statistical Summary of Power & Energy

Calculated across all valid recorded hours ($N = 37,882$):

| Metric | Power Mean (`power`, W) | Power Mean (kW) | Hourly Energy (`energy_kwh`, kWh) |
| :--- | :--- | :--- | :--- |
| **Mean** | 27,791.46 W | 27.79 kW | **27.74 kWh** |
| **Std Dev** | 13,778.19 W | 13.78 kW | **13.78 kWh** |
| **Min** | 0.00 W | 0.00 kW | **0.00 kWh** (Preserved blackouts) |
| **25th Percentile** | 18,439.26 W | 18.44 kW | **18.42 kWh** |
| **Median (50%)** | 23,284.38 W | 23.28 kW | **23.26 kWh** |
| **75th Percentile** | 32,917.61 W | 32.92 kW | **32.87 kWh** |
| **Max** | 79,747.32 W | 79.75 kW | **79.75 kWh** (Peak summer cooling) |

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
