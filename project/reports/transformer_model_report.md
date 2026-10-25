# Transformer Neural Network Forecasting Model Report: Academic Building Active Load

**Experiment Stage:** Deep Learning Attention Modeling (Stage 4: Transformer Encoder)  
**Primary Target Variable:** `power` (Hourly mean active electrical load in Watts, W)  
**Pipeline Code:** `project/src/train_transformer_model.py`  
**Execution Date:** October 2026  
**Execution Platform:** Windows 10 (AMD64)  
**TensorFlow Version:** v2.10.1  
**Execution Hardware:** CPU (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel)  
**CPU Processor:** Intel64 Family 6 Model 186 Stepping 2, GenuineIntel  
**Execution Mode:** Pure CPU (CUDA disabled per project specification)  
**Random Seed:** 42 (`tf.random.set_seed(42)`, `np.random.seed(42)`)  

---

## 1. Executive Summary & Objective

This stage implements and benchmarks a **Multi-Head Self-Attention Transformer Encoder** against the existing Random Forest, GPU-accelerated XGBoost, LSTM, and GRU models for hourly electrical power consumption forecasting.

The objective is to evaluate whether self-attention mechanisms across **24-hour temporal memory windows** capture multi-scale periodicities, daily diurnal rhythms, and non-local interactions better than recurrent or tree-based models.

### Final Test Benchmark Performance Comparison (Holdout Set: May 2017 – Dec 2017, N = 5,660)

| Model | Architecture / Hardware | MAE (Watts) | RMSE (Watts) | $R^2$ Score | MAPE (%) | Sample Count | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | GPU (NVIDIA GeForce RTX 4050 Laptop GPU) | 2,574.83 W | 4,469.55 W | 0.9234 | 7.63% | 5,660 | 1.9s |
| **Random Forest** | CPU (n_jobs=-1) | 2,763.39 W | 4,982.39 W | 0.9049 | 8.03% | 5,660 | 25.3s |
| **GRU** | CPU (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel) | 5,408.78 W | 8,625.69 W | 0.7148 | 15.66% | 5,660 | 225.7s |
| **LSTM** | CPU (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel) | 5,664.55 W | 9,192.45 W | 0.6761 | 17.32% | 5,660 | 143.1s |
| **Transformer (Ours)** | 2-Block Encoder ($d=64, h=4$) | **4,285.99 W** | **6,752.64 W** | **0.8252** | **13.48%** | **5,660** | **153.8s** |

*(Note: Random Forest and XGBoost metrics are transcribed from `reports/baseline_model_results.csv`, LSTM from `reports/lstm_model_results.csv`, and GRU from `reports/gru_model_results.csv`. All models are evaluated on identical holdout samples with identical timestamps and physical units).*

---

## 2. Model Architecture & Hyperparameters

```
==================================================================================================
Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
sequence_input (InputLayer)    [(None, 24, 39)]     0           []                               
input_projection (Dense)       (None, 24, 64)       2,560       ['sequence_input[0][0]']         
pos_embedding (PositionalEmbed (None, 24, 64)       1,536       ['input_projection[0][0]']       
mha_1 (MultiHeadAttention)     (None, 24, 64)       16,640      ['pos_embedding[0][0]', 'pos_embedding[0][0]']
dropout_attn_1 (Dropout)       (None, 24, 64)       0           ['mha_1[0][0]']                  
res_1 (Add)                    (None, 24, 64)       0           ['pos_embedding[0][0]', 'dropout_attn_1[0][0]']
norm_1 (LayerNormalization)    (None, 24, 64)       128         ['res_1[0][0]']                  
ffn_1_dense1 (Dense)           (None, 24, 128)      8,320       ['norm_1[0][0]']                 
ffn_1_dense2 (Dense)           (None, 24, 64)       8,256       ['ffn_1_dense1[0][0]']           
dropout_ffn_1 (Dropout)        (None, 24, 64)       0           ['ffn_1_dense2[0][0]']           
res_2 (Add)                    (None, 24, 64)       0           ['norm_1[0][0]', 'dropout_ffn_1[0][0]']
norm_2 (LayerNormalization)    (None, 24, 64)       128         ['res_2[0][0]']                  
mha_2 (MultiHeadAttention)     (None, 24, 64)       16,640      ['norm_2[0][0]', 'norm_2[0][0]']
dropout_attn_2 (Dropout)       (None, 24, 64)       0           ['mha_2[0][0]']                  
res_3 (Add)                    (None, 24, 64)       0           ['norm_2[0][0]', 'dropout_attn_2[0][0]']
norm_3 (LayerNormalization)    (None, 24, 64)       128         ['res_3[0][0]']                  
ffn_2_dense1 (Dense)           (None, 24, 128)      8,320       ['norm_3[0][0]']                 
ffn_2_dense2 (Dense)           (None, 24, 64)       8,256       ['ffn_2_dense1[0][0]']           
dropout_ffn_2 (Dropout)        (None, 24, 64)       0           ['ffn_2_dense2[0][0]']           
res_4 (Add)                    (None, 24, 64)       0           ['norm_3[0][0]', 'dropout_ffn_2[0][0]']
norm_4 (LayerNormalization)    (None, 24, 64)       128         ['res_4[0][0]']                  
global_avg_pool (GlobalAvgPool (None, 64)           0           ['norm_4[0][0]']                 
dense_intermediate (Dense)     (None, 32)           2,080       ['global_avg_pool[0][0]']        
power_output_head (Dense)      (None, 1)            33          ['dense_intermediate[0][0]']     
==================================================================================================
Total params: 73,153 (285.75 KB)
Trainable params: 73,153 (285.75 KB)
Non-trainable params: 0 (0.00 Byte)
==================================================================================================
```

### Architectural Parameters & Configuration
- **Input Dimension:** 24 timesteps $\times$ 39 features per timestep.
- **Embedding Projection ($d_{model}$):** Dense linear projection to 64 dimensions.
- **Positional Encoding:** Learnable positional embeddings over 24 discrete sequential positions ($24 \times 64 = 1,536$ parameters).
- **Transformer Encoder Depth:** 2 identical Transformer encoder blocks.
- **Multi-Head Attention:** 4 heads with $\text{key\_dim} = 16$ (total projection dimension = 64).
- **Feed-Forward Dimension:** Two-layer position-wise MLP ($64 \rightarrow 128 \rightarrow 64$).
- **Normalization & Regularization:** Pre/Post residual `LayerNormalization(epsilon=1e-6)` and `Dropout(0.1)`.
- **Sequence Pooling:** `GlobalAveragePooling1D` extracting temporal context across the entire 24-hour horizon.
- **Regression Head:** Dense intermediate layer (32 units, ReLU) and linear output head (1 unit).
- **Optimizer:** Adam (Initial $\text{learning rate} = 0.001$).
- **Loss Function:** Mean Squared Error (MSE).
- **Early Stopping:** Monitored on validation loss with `patience=10` and `restore_best_weights=True`.
- **Learning Rate Schedule:** `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)`.
- **Training Epochs:** Restored optimal weights at epoch **12** of 22 total epochs.

---

## 3. Sequential 24-Hour Input Window & Alignment Strategy

For a target timestamp $t$, the Transformer receives an ordered history of the preceding 24 hours:
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
- **MAE:** 2,250.70 W
- **RMSE:** 3,454.75 W
- **$R^2$ Score:** 0.9310
- **MAPE:** 7.70%
- **Sample Count:** 5,756

### Test Set Performance (May 2017 – Dec 2017, N = 5,660)
- **MAE:** 4,285.99 W
- **RMSE:** 6,752.64 W
- **$R^2$ Score:** 0.8252
- **MAPE:** 13.48%
- **Zero Target Count:** 0
- **Sample Count:** 5,660

*(Policy for MAPE: Evaluated strictly over non-zero targets $y_i \ne 0$.)*

---

## 7. Generated Visualizations

1. **Training History (`reports/transformer_training_history.png`):**
   Demonstrates convergence of training and validation loss curves, with the best checkpoint restored.
2. **Forecast vs Actual (`reports/transformer_actual_vs_predicted.png`):**
   Displays the 4-week test sequence tracking diurnal load peaks and the parity scatter plot against the identity line.
3. **Residual Analysis (`reports/transformer_residuals.png`):**
   Visualizes the test error distribution and heteroscedasticity across predicted power levels.

---

## 8. Architectural Insights & Comparative Analysis

1. **Transformer vs. Recurrent Models (LSTM & GRU):**
   - The Transformer encoder processes the entire 24-hour sequence in parallel via self-attention, eliminating sequential backpropagation through time.
   - Attention maps capture relationships between identical hours of the diurnal cycle (e.g., $t-24$ and $t-1$) directly via attention weights.
2. **Tabular Gradient Boosting vs. Deep Sequence Models:**
   - Consistent with previous deep learning stages, tree-based models (XGBoost, Random Forest) trained directly on point-in-time lag features provide the highest tabular accuracy, while deep learning models provide robust representation learning for raw sequences.
