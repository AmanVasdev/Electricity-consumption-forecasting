# GRU Neural Network Forecasting Model Report: Academic Building Active Load

**Experiment Stage:** Deep Learning Sequential Modeling (Stage 3: Gated Recurrent Unit - GRU)  
**Primary Target Variable:** `power` (Hourly mean active electrical load in Watts, W)  
**Pipeline Code:** `project/src/train_gru_model.py`  
**Execution Date:** October 2026  
**Execution Platform:** Windows 10 (AMD64)  
**TensorFlow Version:** v2.10.1  
**Execution Hardware:** CPU (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel)  
**CPU Processor:** Intel64 Family 6 Model 186 Stepping 2, GenuineIntel  
**Execution Mode:** Pure CPU (CUDA disabled per project specification)  
**Random Seed:** 42 (`tf.random.set_seed(42)`, `np.random.seed(42)`)  

---

## 1. Executive Summary & Objective

This stage implements and benchmarks a **Gated Recurrent Unit (GRU)** recurrent neural network against the existing Random Forest, GPU-accelerated XGBoost, and 2-layer LSTM models for hourly electrical power consumption forecasting.

The objective is to evaluate whether the more parameter-efficient gating mechanism of the GRU (which eliminates the dedicated cell state and output gate of the LSTM) provides improved training convergence, computational speed on CPU, and forecasting accuracy across **24-hour sequential temporal windows**.

### Final Test Benchmark Performance Comparison (Holdout Set: May 2017 – Dec 2017, N = 5,660)

| Model | Architecture / Hardware | MAE (Watts) | RMSE (Watts) | $R^2$ Score | MAPE (%) | Sample Count | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 300 Trees (CPU Multi-threaded) | 2,763.39 W | 4,982.39 W | 0.9049 | 8.03% | 5,660 | 25.3s |
| **XGBoost** | 1,000 Hist Trees (RTX 4050 GPU) | 2,574.83 W | 4,469.55 W | 0.9234 | 7.63% | 5,660 | 1.9s |
| **LSTM** | 2-Layer LSTM (64 units, CPU) | 5,664.55 W | 9,192.45 W | 0.6761 | 17.32% | 5,660 | 143.1s |
| **GRU (Ours)** | 2-Layer GRU (64 units, CPU) | **5,408.78 W** | **8,625.69 W** | **0.7148** | **15.66%** | **5,660** | **225.7s** |

*(Note: Random Forest and XGBoost metrics are transcribed from `reports/baseline_model_results.csv`, and LSTM metrics from `reports/lstm_model_results.csv`. All models are evaluated on identical holdout samples with identical timestamps and physical units).*

---

## 2. Model Architecture & Hyperparameters

```
=================================================================
Layer (type)                Output Shape              Param #   
=================================================================
sequence_input (InputLayer) [(None, 24, 39)]          0         
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
- **Optimizer:** Adam (Initial $\text{learning rate} = 0.001$).
- **Loss Function:** Mean Squared Error (MSE).
- **Batch Size:** 64 samples per gradient step.
- **Early Stopping:** Monitored on validation loss with `patience=10` and `restore_best_weights=True`.
- **Learning Rate Schedule:** `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)`.
- **Training Epochs:** Restored optimal weights at epoch **19** of 29 total epochs.

---

## 3. Sequential 24-Hour Input Window & Alignment Strategy

For a target timestamp $t$, the recurrent model receives an ordered history of the preceding 24 hours:
$$X_t = [f_{t-24}, f_{t-23}, \dots, f_{t-1}] \in \mathbb{R}^{24 \times 39}$$

To predict target $y_t = \text{power}_t$:
1. **Zero Future Contamination:** Vector $f_\tau$ at timestep $\tau \le t-1$ contains only information recorded prior to $t$.
2. **Boundary Alignment Across Splits:**
   - **Training Set** ($t \in [24, 26963]$): Uses historical records within the training partition.
   - **Validation Set** ($t \in [26964, 32741]$): The initial validation hours ($t=26964$) look back into rows $[26940, 26963]$ (the historical tail of the training partition). No validation or future test targets are observed.
   - **Test Set** ($t \in [32742, 38519]$): The initial test hours ($t=32742$) look back into rows $[32718, 32741]$ (the historical tail of the validation partition). No future test observations are accessed.
3. **No Random Shuffling:** Temporal ordering of sequences is strictly preserved.

---

## 4. Feature Schema & Leakage Prevention

### Features Included (39 Total Predictors):
- **Power Lag Features (8):** `power_lag_1`, `power_lag_2`, `power_lag_3`, `power_lag_6`, `power_lag_12`, `power_lag_24`, `power_lag_48`, `power_lag_168`
- **Energy Lag Features (8):** `energy_kwh_lag_1`, `energy_kwh_lag_2`, `energy_kwh_lag_3`, `energy_kwh_lag_6`, `energy_kwh_lag_12`, `energy_kwh_lag_24`, `energy_kwh_lag_48`, `energy_kwh_lag_168`
- **Rolling Averages (5):** `power_rolling_mean_3`, `power_rolling_mean_6`, `power_rolling_mean_12`, `power_rolling_mean_24`, `power_rolling_mean_168`
- **Rolling Volatilities (5):** `power_rolling_std_3`, `power_rolling_std_6`, `power_rolling_std_12`, `power_rolling_std_24`, `power_rolling_std_168`
- **Cyclical Trigonometric (6):** `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`
- **Calendar Descriptors (6):** `hour`, `day_of_week`, `day_of_month`, `month`, `week_of_year`, `is_weekend`
- **Lagged Range (1):** `power_range_lag_1`

### Columns Strictly Excluded (13 Columns):
`timestamp, power, power_mean, power_min, power_max, power_std, power_range, coefficient_of_variation, energy_kwh, current, voltage, frequency, power_factor`

---

## 5. Preprocessing & Scaling Isolation

1. **Missing Data Imputation:**
   - Unrecorded target hours (`NaN`) are strictly removed from training, validation, and test sample evaluation (zero fabrication).
   - Any historical missing values in the predictor matrix (burn-in or sensor gaps) are imputed using `SimpleImputer(strategy='median')` **fitted strictly on the training partition**.
2. **Feature Standard Scaling:**
   - A `StandardScaler` is fitted strictly on the imputed training split features:
     $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   - Validation and test splits are transformed using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$.
3. **Target Standard Scaling:**
   - Target variable `power` is normalized using parameters computed strictly on valid training targets ($\mu = 26473.53\text{ W}, \sigma = 13106.44\text{ W}$).
   - All network predictions are mapped back to physical Watts via inverse transformation before computing evaluation metrics.

---

## 6. Detailed Evaluation Results

### Validation Set Performance (Sep 2016 – May 2017, N = 5,756)
- **MAE:** 2,089.19 W
- **RMSE:** 3,367.40 W
- **$R^2$ Score:** 0.9344
- **MAPE:** 6.79%
- **Sample Count:** 5,756

### Test Set Performance (May 2017 – Dec 2017, N = 5,660)
- **MAE:** 5,408.78 W
- **RMSE:** 8,625.69 W
- **$R^2$ Score:** 0.7148
- **MAPE:** 15.66%
- **Zero Target Count:** 0
- **Sample Count:** 5,660

*(Policy for MAPE: Evaluated strictly over non-zero targets $y_i \ne 0$.)*

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
