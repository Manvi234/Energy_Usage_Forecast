import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ── Load assets ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    pt        = joblib.load('pt_transformer.pkl')
    scaler    = joblib.load('scaler.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    lgb_model = joblib.load('lgb_model.pkl')
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return pt, scaler, xgb_model, lgb_model, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv('df_featured.csv', index_col=0, parse_dates=True)
    results = pd.read_csv('model_results.csv')
    return df, results

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

@st.cache_resource
def load_lstm(_scaler):
    model = LSTMForecaster()
    model.load_state_dict(torch.load('lstm_weights.pth',
                                      map_location='cpu'))
    model.eval()
    return model

# ── Load everything ──────────────────────────────────────────
pt, scaler, xgb_model, lgb_model, feature_cols = load_models()
df, results_df = load_data()
lstm_model = load_lstm(scaler)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("⚡ Energy Forecast")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Model Comparison", "🔮 Forecast", "📈 Data Explorer"]
)

# ════════════════════════════════════════════════════════════
# PAGE 1 — Model Comparison
# ════════════════════════════════════════════════════════════
if page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")
    st.markdown("Benchmarking 4 models on UCI Appliances Energy dataset")

    # Metrics table
    st.subheader("Results Table")
    st.dataframe(
        results_df.style.highlight_min(
            subset=['MAPE(>50W)', 'RMSE', 'MAE'],
            color='lightgreen'
        ),
        use_container_width=True
    )

    # Bar charts
    st.subheader("Visual Comparison")
    col1, col2, col3 = st.columns(3)

    metrics = [
        ('MAPE(>50W)', 'MAPE % (>50W)', col1, 'steelblue'),
        ('RMSE',       'RMSE (Watts)',   col2, 'crimson'),
        ('MAE',        'MAE (Watts)',    col3, 'green'),
    ]

    for metric, ylabel, col, color in metrics:
        with col:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(results_df['Model'], results_df[metric],
                   color=color, alpha=0.7)
            ax.set_title(metric)
            ax.set_ylabel(ylabel)
            ax.set_xticklabels(results_df['Model'],
                               rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)

    # Key insights
    st.subheader("Key Findings")
    st.info("📌 ARIMA → XGBoost: 18% MAPE reduction through feature engineering")
    st.info("📌 XGBoost ≈ LightGBM: Tree models converged at data ceiling (~22%)")
    st.info("📌 LSTM wins on RMSE (55.49W) — better on large spike prediction")
    st.info("📌 Optuna tuning reduced XGBoost MAPE from 153% → 22.3%")

# ════════════════════════════════════════════════════════════
# PAGE 2 — Forecast
# ════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.title("🔮 Live Forecast")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")
        model_choice = st.selectbox(
            "Select Model",
            ["XGBoost", "LightGBM", "LSTM"]
        )
        n_steps = st.slider(
            "Forecast steps to display",
            min_value=50, max_value=500, value=200, step=50
        )

        st.markdown("---")
        st.markdown("**Model Info**")
        info = {
            "XGBoost":  ("22.33%", "57.66W", "Optuna tuned, 500 trees"),
            "LightGBM": ("22.48%", "57.07W", "Optuna tuned, leaf-wise"),
            "LSTM":     ("22.60%", "55.49W", "2-layer, seq_len=144"),
        }
        mape, rmse, desc = info[model_choice]
        st.metric("MAPE (>50W)", mape)
        st.metric("RMSE", rmse)
        st.caption(desc)

    with col2:
        st.subheader(f"{model_choice} — Actual vs Predicted")

        # Get test data
        split_point = int(len(df) * 0.8)
        test_df = df.iloc[split_point:]
        X_test  = test_df[feature_cols]
        y_test  = test_df['Appliances'].values

        # Generate predictions
        if model_choice == "XGBoost":
            preds = xgb_model.predict(X_test)

        elif model_choice == "LightGBM":
            preds = lgb_model.predict(X_test)

        else:
            # LSTM
            test_scaled = scaler.transform(test_df[['Appliances']])
            SEQ_LEN = 144
            X_seq, y_seq = [], []
            for i in range(len(test_scaled) - SEQ_LEN):
                X_seq.append(test_scaled[i:i+SEQ_LEN])
                y_seq.append(test_scaled[i+SEQ_LEN])
            X_tensor = torch.FloatTensor(np.array(X_seq))
            with torch.no_grad():
                lstm_out = lstm_model(X_tensor).numpy()
            preds  = scaler.inverse_transform(lstm_out).flatten()
            y_test = scaler.inverse_transform(
                        np.array(y_seq).reshape(-1, 1)).flatten()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test[:n_steps], label='Actual',
                color='steelblue', alpha=0.8)
        ax.plot(preds[:n_steps],  label=f'{model_choice} Predicted',
                color='orange', alpha=0.7)
        ax.set_title(f'{model_choice} — First {n_steps} test steps')
        ax.set_ylabel('Appliance Usage (Watts)')
        ax.set_xlabel('Time Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

# ════════════════════════════════════════════════════════════
# PAGE 3 — Data Explorer
# ════════════════════════════════════════════════════════════
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Consumption by Hour")
        hourly = df.groupby(df.index.hour)['Appliances'].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(hourly.index, hourly.values, color='teal', alpha=0.7)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Watts')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Consumption by Day of Week")
        daily = df.groupby(df.index.dayofweek)['Appliances'].mean()
        days  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(days, daily.values, color='coral', alpha=0.7)
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Mean Watts')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Raw Consumption Over Time")
    daily_avg = df['Appliances'].resample('D').mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_avg, color='steelblue', alpha=0.8)
    ax.set_ylabel('Mean Daily Watts')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Raw Data Sample")
    st.dataframe(
        df[['Appliances', 'T_out', 'RH_out',
            'hour', 'day_of_week']].tail(20),
        use_container_width=True
    )
