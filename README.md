# ⚡ Energy Consumption Forecasting — Progressive Model Stack

A end-to-end machine learning project benchmarking 4 forecasting models on household appliance energy consumption data. The project progressively improves from a statistical baseline to deep learning, with each model building on insights from the last.

**[🚀 Live Dashboard → energyusageforecast.streamlit.app](https://energyusageforecast.streamlit.app)**

---

## Results

| Model | MAPE (>50W) | RMSE | MAE |
|---|---|---|---|
| ARIMA(1,0,0) | 27.40% | 94.53W | 43.22W |
| XGBoost (Optuna) | 22.33% | 57.66W | 24.95W |
| LightGBM (Optuna) | 22.48% | 57.07W | 24.86W |
| **LSTM** | **22.60%** | **55.49W** | **24.83W** |

> MAPE computed on active consumption readings (>50W) to avoid mathematical distortion from standby readings. LSTM achieves best absolute error metrics (RMSE, MAE) despite similar MAPE to tree models.

---

## Project Structure

```
energy-forecast/
├── notebook.ipynb          # Full pipeline — EDA through modeling
├── dashboard.py            # Streamlit app
├── data/
│   └── energydata_complete.csv
├── models/
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── lstm_weights.pth
│   ├── scaler.pkl
│   └── pt_transformer.pkl
└── results/
    └── model_comparison.csv
```

---

## Dataset

**UCI Appliances Energy Prediction**
- 19,735 readings at 10-minute intervals (January–May 2016)
- 1 Belgian household, 28 features including indoor/outdoor temperature, humidity, and weather
- Target: `Appliances` — total appliance energy consumption in Watt-hours

[Download from UCI Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

---

## Methodology

### 1. Exploratory Data Analysis
- STL decomposition confirmed daily (144-step) and weekly seasonal cycles
- ACF/PACF identified AR(1) process after Yeo-Johnson transformation
- ADF test confirmed stationarity post-transformation (p=0.000)
- `lights` column identified as zero-inflated (77% zeros) — re-engineered as binary `lights_on` feature
- `rv1`, `rv2` confirmed as noise columns (correlation < 0.05) and excluded

### 2. Feature Engineering
Built 50 features from raw time series:
- **Lag features:** t-1, t-2, t-3, t-6, t-12, t-24, t-144 (24h ago)
- **Rolling statistics:** mean and std over 3, 6, 12, 24, 144 window sizes
- **Calendar features:** hour, day_of_week, is_weekend, month
- **Weather interactions:** temp × humidity, temp², lagged weather

> Tested rate-of-change features (diff_1, diff_3, momentum) — found redundant with existing lag features, excluded from final set.

### 3. Models

**ARIMA(1,0,0)** — Statistical baseline
- Parameters selected via ADF test (d=0), PACF (p=1), ACF (q=0)
- Fit on last 30 days of training data (computational constraint)
- Predicts mean consumption level, misses spikes entirely

**XGBoost + Optuna** — First major improvement
- 50-trial Optuna search optimizing MAE
- Depth reduced from default 6 → optimal 3, preventing overfitting on spike patterns
- MAPE reduced from 153% (default) → 22.33% (tuned)

**LightGBM + Optuna** — Marginal improvement over XGBoost
- Leaf-wise growth with `num_leaves` tuning via Optuna
- Near-identical performance to XGBoost — tree models reached data ceiling

**LSTM** — Best absolute error
- 2-layer LSTM, hidden_size=64, dropout=0.2
- Sequence length=144 (24h lookback), justified by ACF analysis in EDA
- RobustScaler on target, seed=42 for reproducibility
- Outperforms tree models on RMSE and MAE despite similar MAPE

### 4. Key Findings

1. **Feature engineering drove the largest gain** — ARIMA→XGBoost (18% MAPE reduction) from calendar and lag features, not model complexity
2. **Optuna tuning was critical** — default XGBoost hyperparameters gave 153% MAPE; tuning brought it to 22.33%
3. **Tree models and LSTM converged** — all three reached ~22-23% MAPE, indicating a data ceiling from unpredictable consumption spikes
4. **LSTM wins on spike prediction** — lower RMSE (55.49W vs 57W+) shows better handling of large consumption events
5. **Multivariate LSTM underperformed** — adding weather/calendar features worsened LSTM performance; fixed hidden size spread too thin across 6 input features

---

## Installation

```bash
git clone https://github.com/yourusername/energy-forecast
cd energy-forecast
pip install -r requirements.txt
streamlit run dashboard.py
```

**requirements.txt**
```
streamlit
pandas
numpy
matplotlib
scikit-learn==1.6.1
xgboost
lightgbm
torch
optuna
statsmodels
seaborn
joblib
```

---

## Dashboard

The Streamlit dashboard has three pages:

- **📊 Model Comparison** — metrics table and bar charts across all 4 models
- **🔮 Forecast** — select any model and forecast horizon, view actual vs predicted
- **📈 Data Explorer** — hourly/daily consumption patterns, raw data sample

---

## Tech Stack

`Python` `PyTorch` `XGBoost` `LightGBM` `Optuna` `statsmodels` `scikit-learn` `Streamlit` `Pandas` `NumPy` `Matplotlib` `Seaborn`

---

## Resume Summary

*Benchmarked 4-model forecasting stack (ARIMA → XGBoost → LightGBM → LSTM) on UCI Appliances Energy dataset; reduced MAPE from 27.4% → 22.6% and RMSE from 94.5W → 55.5W through progressive feature engineering, Optuna hyperparameter tuning, and sequence modeling; deployed interactive forecast dashboard at energyusageforecast.streamlit.app*
