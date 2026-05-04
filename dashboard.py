import lightgbm  # must precede torch on macOS to avoid segfault
import xgboost    # same
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import joblib
import pickle
import torch
import torch.nn as nn
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

BASE = Path(__file__).parent

st.set_page_config(
    page_title="Energy Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0d14;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"]  { background: transparent; }

/* Chapter dividers */
.chapter-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 4px;
}
.chapter-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    color: #f1f5f9;
    line-height: 1.2;
    margin-bottom: 10px;
}
.chapter-body {
    font-size: 1rem;
    color: #94a3b8;
    line-height: 1.8;
    max-width: 680px;
    margin-bottom: 32px;
}

/* Hero */
.hero-wrap {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 60px 48px;
    margin-bottom: 48px;
    border: 1px solid #1e293b;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #f8fafc;
    line-height: 1.15;
    margin-bottom: 18px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 580px;
    line-height: 1.75;
    margin-bottom: 36px;
}

/* Stat pills */
.stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }
.stat-pill {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 8px 20px;
    font-size: 0.85rem;
    color: #cbd5e1;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.stat-pill strong { color: #f1f5f9; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 20px 24px;
}
[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f9fafb !important;
    font-size: 1.7rem !important;
    font-weight: 700;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* Section separator */
.section-sep {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 48px 0;
}

/* Insight callout */
.callout {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.7;
    margin-bottom: 12px;
}
.callout strong { color: #e2e8f0; }

/* Model badge */
.model-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #2563eb33;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
}

/* Footer */
.footer {
    text-align: center;
    color: #374151;
    font-size: 0.78rem;
    padding: 40px 0 24px;
    border-top: 1px solid #111827;
    margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────
PL = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#94a3b8", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#1f2937", showline=False, tickfont=dict(color="#6b7280")),
    yaxis=dict(gridcolor="#1f2937", showline=False, tickfont=dict(color="#6b7280")),
    margin=dict(l=8, r=8, t=44, b=8),
    legend=dict(bgcolor="#1f2937", bordercolor="#374151", borderwidth=1,
                font=dict(color="#d1d5db")),
)

# ── Model ─────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        pt        = joblib.load(BASE / 'pt_transformer.pkl')
        scaler    = joblib.load(BASE / 'scaler.pkl')
        xgb_model = joblib.load(BASE / 'xgb_model.pkl')
        lgb_model = joblib.load(BASE / 'lgb_model.pkl')
        with open(BASE / 'feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return pt, scaler, xgb_model, lgb_model, feature_cols
    except Exception as e:
        st.error(f"Failed to load models: {e}"); st.stop()

@st.cache_data(show_spinner=False)
def load_data():
    try:
        df      = pd.read_csv(BASE / 'df_featured.csv', index_col=0, parse_dates=True)
        results = pd.read_csv(BASE / 'model_results.csv')
        return df, results
    except Exception as e:
        st.error(f"Failed to load data: {e}"); st.stop()

@st.cache_resource(show_spinner=False)
def load_lstm():
    model = LSTMForecaster()
    model.load_state_dict(torch.load(BASE / 'lstm_weights.pth', map_location='cpu'))
    model.eval()
    return model

with st.spinner("Loading…"):
    pt, scaler, xgb_model, lgb_model, feature_cols = load_models()
    df, results_df = load_data()
    lstm_model = load_lstm()

# ════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════
avg_w   = df['Appliances'].mean()
peak_w  = df['Appliances'].max()
n_days  = (df.index.max() - df.index.min()).days

st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-eyebrow">Energy Forecasting</div>
  <div class="hero-title">Can we predict<br>home energy use?</div>
  <div class="hero-sub">
    A Belgian smart home logged every appliance in 10-minute intervals
    for {n_days} days. We took that data, built four forecasting models,
    and asked: <em>how accurately can we predict what a household will consume next?</em>
  </div>
  <div class="stat-row">
    <div class="stat-pill">🗓 <strong>{n_days} days</strong> of data</div>
    <div class="stat-pill">⚡ <strong>{df.shape[0]:,}</strong> readings</div>
    <div class="stat-pill">📊 <strong>{df.shape[1]}</strong> features</div>
    <div class="stat-pill">🔥 Peak <strong>{peak_w:.0f} W</strong></div>
    <div class="stat-pill">〜 Avg <strong>{avg_w:.0f} W</strong></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CHAPTER 1 — THE DATA
# ════════════════════════════════════════════════════════════
st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
st.markdown("""
<div class="chapter-label">Chapter 01</div>
<div class="chapter-title">Getting to know the data</div>
<div class="chapter-body">
Before building any model, we explored the patterns hidden in the readings.
Energy consumption isn't random. It follows the rhythms of daily life.
Mornings spike, nights dip. Weekdays and weekends look completely different.
The charts below tell that story.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    hourly = df.groupby(df.index.hour)['Appliances'].mean().reset_index()
    hourly.columns = ['Hour', 'Watts']
    peak_hour = hourly.loc[hourly['Watts'].idxmax(), 'Hour']
    fig = go.Figure(go.Bar(
        x=hourly['Hour'], y=hourly['Watts'],
        marker=dict(
            color=hourly['Watts'],
            colorscale=[[0, '#1e3a5f'], [1, '#3b82f6']],
            line_width=0,
        ),
        hovertemplate="<b>%{x}:00</b><br>Avg: %{y:.1f} W<extra></extra>",
    ))
    fig.update_layout(**PL,
        title=dict(text=f"Consumption by Hour  ·  peak at {peak_hour}:00", font=dict(color="#e2e8f0", size=13)),
        xaxis_title="Hour of Day", yaxis_title="Mean Watts", height=300, bargap=0.25)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    daily = df.groupby(df.index.dayofweek)['Appliances'].mean()
    days  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    peak_day = days[daily.idxmax()]
    fig = go.Figure(go.Bar(
        x=days, y=daily.values,
        marker=dict(
            color=daily.values,
            colorscale=[[0, '#3b1f5e'], [1, '#a855f7']],
            line_width=0,
        ),
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f} W<extra></extra>",
    ))
    fig.update_layout(**PL,
        title=dict(text=f"Consumption by Day  ·  peak on {peak_day}", font=dict(color="#e2e8f0", size=13)),
        xaxis_title="Day of Week", yaxis_title="Mean Watts", height=300, bargap=0.35)
    st.plotly_chart(fig, use_container_width=True)

# Timeline
daily_avg = df['Appliances'].resample('D').mean().reset_index()
daily_avg.columns = ['Date', 'Watts']
mean_line = daily_avg['Watts'].mean()
fig = go.Figure()
fig.add_hrect(y0=mean_line * 0.93, y1=mean_line * 1.07,
              fillcolor="#1d4ed8", opacity=0.07, line_width=0,
              annotation_text="±7% of mean", annotation_position="top left",
              annotation_font=dict(color="#64748b", size=11))
fig.add_trace(go.Scatter(
    x=daily_avg['Date'], y=daily_avg['Watts'],
    fill='tozeroy', fillcolor='rgba(59,130,246,0.07)',
    line=dict(color='#3b82f6', width=2),
    hovertemplate="<b>%{x|%b %d}</b><br>%{y:.1f} W<extra></extra>",
))
fig.update_layout(**PL,
    title=dict(text="Daily Mean Consumption  ·  Jan – May 2016", font=dict(color="#e2e8f0", size=13)),
    xaxis_title="", yaxis_title="Mean Watts", height=260, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="callout">
  📌 <strong>Key observation:</strong> Consumption is highly periodic with strong hourly and
  day-of-week seasonality, a good sign that machine learning models can learn these patterns.
  The winter-to-spring transition in the data also introduces a gradual downward trend worth capturing.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CHAPTER 2 — THE MODELS
# ════════════════════════════════════════════════════════════
st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
st.markdown("""
<div class="chapter-label">Chapter 02</div>
<div class="chapter-title">Four models, one question</div>
<div class="chapter-body">
We didn't just train one model and call it a day. We built a progression,
starting from a classical statistical baseline and working up to deep learning,
to understand exactly where the gains come from.
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
model_cards = [
    (c1, "ARIMA",    "The baseline",     "Classical stats, no feature engineering. Fast to train, limited ceiling.",              "#ef4444"),
    (c2, "XGBoost",  "The workhorse",    "Gradient boosted trees with 29 engineered features and Optuna hyperparameter search.",  "#f97316"),
    (c3, "LightGBM", "The challenger",   "Leaf-wise boosting, nearly identical to XGBoost but slightly faster to train.",        "#a855f7"),
    (c4, "LSTM",     "The deep learner", "2-layer recurrent network trained on raw sequences. Excels at capturing spikes.",       "#3b82f6"),
]
for col, name, tagline, desc, color in model_cards:
    with col:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2937;border-top:3px solid {color};
                    border-radius:10px;padding:20px 18px;height:180px;">
          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
                      text-transform:uppercase;color:{color};margin-bottom:6px">{tagline}</div>
          <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;margin-bottom:10px">{name}</div>
          <div style="font-size:0.82rem;color:#6b7280;line-height:1.6">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CHAPTER 3 — THE RESULTS
# ════════════════════════════════════════════════════════════
st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
st.markdown("""
<div class="chapter-label">Chapter 03</div>
<div class="chapter-title">The verdict</div>
<div class="chapter-body">
After training, tuning, and evaluating on a held-out 20% test set,
here's how the four models stack up. Lower is better on all three metrics.
</div>
""", unsafe_allow_html=True)

# Summary metrics
best_mape = results_df.loc[results_df['MAPE(>50W)'].idxmin()]
best_rmse = results_df.loc[results_df['RMSE'].idxmin()]
worst     = results_df.loc[results_df['RMSE'].idxmax()]
improvement = ((worst['RMSE'] - best_rmse['RMSE']) / worst['RMSE']) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Best MAPE",        f"{best_mape['MAPE(>50W)']:.2f}%",  f"{best_mape['Model']}")
m2.metric("Best RMSE",        f"{best_rmse['RMSE']:.2f} W",       f"{best_rmse['Model']}")
m3.metric("RMSE improvement", f"{improvement:.0f}%",              "vs. ARIMA baseline")
m4.metric("Models evaluated", f"{len(results_df)}", "full pipeline")

st.markdown("<br>", unsafe_allow_html=True)

# Charts
col1, col2, col3 = st.columns(3)
chart_cfg = [
    ('MAPE(>50W)', 'MAPE % (>50W)', '#3b82f6', col1),
    ('RMSE',       'RMSE (Watts)',   '#f43f5e', col2),
    ('MAE',        'MAE  (Watts)',   '#10b981', col3),
]
for metric, label, color, col in chart_cfg:
    colors = [color if v == results_df[metric].min() else '#1f2937'
              for v in results_df[metric]]
    borders = [color if v == results_df[metric].min() else '#374151'
               for v in results_df[metric]]
    fig = go.Figure(go.Bar(
        x=results_df['Model'], y=results_df[metric],
        marker=dict(color=colors, line=dict(color=borders, width=2)),
        hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(**PL,
        title=dict(text=f"{metric}  <span style='color:#374151;font-size:11px'>· best highlighted</span>",
                   font=dict(color="#e2e8f0", size=13)),
        yaxis_title=label, height=300, bargap=0.4)
    col.plotly_chart(fig, use_container_width=True)

# Table
st.markdown("<br>", unsafe_allow_html=True)
st.dataframe(
    results_df.style
        .highlight_min(subset=['MAPE(>50W)', 'RMSE', 'MAE'], color='#14532d')
        .format({'MAPE(>50W)': '{:.2f}%', 'RMSE': '{:.2f} W', 'MAE': '{:.2f} W'}),
    use_container_width=True, height=175,
)

st.markdown("""
<div class="callout" style="margin-top:20px">
  📌 <strong>The surprise:</strong> XGBoost and LightGBM converged to nearly identical MAPE (~22.3–22.5%).
  The real winner depends on the use case. If you care about peak prediction, LSTM's lower RMSE
  (55.49 W) makes it the better choice. If you need fast inference, XGBoost wins.
</div>
<div class="callout">
  📌 <strong>The biggest jump:</strong> Feature engineering alone (ARIMA → XGBoost) cut RMSE by
  <strong>41%</strong>, far more than any model architecture choice.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CHAPTER 4 — THE FORECAST
# ════════════════════════════════════════════════════════════
st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
st.markdown("""
<div class="chapter-label">Chapter 04</div>
<div class="chapter-title">See it for yourself</div>
<div class="chapter-body">
Pick any model and watch it predict against real test data. The dotted line is
what the model thought would happen. The solid line is what actually did.
</div>
""", unsafe_allow_html=True)

ctrl1, ctrl2, _ = st.columns([1, 1, 2])
with ctrl1:
    model_choice = st.selectbox("Model", ["XGBoost", "LightGBM", "LSTM"], label_visibility="collapsed")
with ctrl2:
    n_steps = st.slider("Steps to display", 50, 500, 200, 50, label_visibility="collapsed")

# Derive predictions
split_point = int(len(df) * 0.8)
test_df = df.iloc[split_point:]
X_test  = test_df[feature_cols]
y_test  = test_df['Appliances'].values

with st.spinner(f"Running {model_choice}…"):
    if model_choice == "XGBoost":
        preds = xgb_model.predict(X_test)
    elif model_choice == "LightGBM":
        preds = lgb_model.predict(X_test)
    else:
        test_scaled = scaler.transform(test_df[['Appliances']])
        SEQ_LEN = 144
        X_seq, y_seq = [], []
        for i in range(len(test_scaled) - SEQ_LEN):
            X_seq.append(test_scaled[i:i + SEQ_LEN])
            y_seq.append(test_scaled[i + SEQ_LEN])
        X_tensor = torch.FloatTensor(np.array(X_seq))
        with torch.no_grad():
            lstm_out = lstm_model(X_tensor).numpy()
        preds  = scaler.inverse_transform(lstm_out).flatten()
        y_test = scaler.inverse_transform(np.array(y_seq).reshape(-1, 1)).flatten()

n     = min(n_steps, len(preds), len(y_test))
x_ax  = list(range(n))
mae_w = np.mean(np.abs(preds[:n] - y_test[:n]))
rmse_w= np.sqrt(np.mean((preds[:n] - y_test[:n]) ** 2))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_ax, y=y_test[:n], name="Actual",
    line=dict(color="#38bdf8", width=2), opacity=0.9,
))
fig.add_trace(go.Scatter(
    x=x_ax, y=preds[:n], name=f"{model_choice} Predicted",
    line=dict(color="#fb923c", width=1.8, dash="dot"), opacity=0.85,
))
fig.update_layout(**PL,
    title=dict(text=f"{model_choice}: Actual vs Predicted  ·  first {n} test steps",
               font=dict(color="#e2e8f0", size=14)),
    xaxis_title="Time Steps (10-min intervals)",
    yaxis_title="Appliance Usage (Watts)",
    height=400, hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

e1, e2, e3 = st.columns(3)
e1.metric("MAE (this window)",  f"{mae_w:.1f} W")
e2.metric("RMSE (this window)", f"{rmse_w:.1f} W")
e3.metric("Time covered",       f"{n * 10 // 60}h {(n * 10) % 60}m")

# ════════════════════════════════════════════════════════════
# EPILOGUE
# ════════════════════════════════════════════════════════════
st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
st.markdown("""
<div class="chapter-label">Epilogue</div>
<div class="chapter-title">What we learned</div>
<div class="chapter-body">
Predicting home energy use is hard. Occupant behaviour is noisy and hard to model.
But structured feature engineering and modern gradient boosting close most of the gap.
The remaining ~22% MAPE represents the irreducible unpredictability of human behaviour.
Future work could explore exogenous signals (weather forecasts, calendar events) or
ensemble methods combining tree models and LSTM.
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.markdown("""
<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:20px 18px">
  <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
              color:#6366f1;margin-bottom:8px">Biggest win</div>
  <div style="font-size:0.95rem;color:#d1d5db;line-height:1.7">
    Feature engineering contributed more accuracy gain than any model upgrade.
    Temporal features (hour, day, lag) were the highest-importance variables across all tree models.
  </div>
</div>
""", unsafe_allow_html=True)

col2.markdown("""
<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:20px 18px">
  <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
              color:#10b981;margin-bottom:8px">Surprising finding</div>
  <div style="font-size:0.95rem;color:#d1d5db;line-height:1.7">
    Optuna tuning reduced XGBoost MAPE from 153% → 22.3%, a 7× improvement
    hyperparameter search matters as much as architecture.
  </div>
</div>
""", unsafe_allow_html=True)

col3.markdown("""
<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:20px 18px">
  <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
              color:#f59e0b;margin-bottom:8px">Next step</div>
  <div style="font-size:0.95rem;color:#d1d5db;line-height:1.7">
    Combining tree models (strong on trend) with LSTM (strong on spikes) into
    a stacking ensemble could push MAPE below 20%.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  UCI Appliances Energy Prediction Dataset · Jan–May 2016 · Belgium<br><br>
  Built by <a href="https://www.linkedin.com/in/manvi-gawande/" target="_blank"
  style="color:#60a5fa;text-decoration:none;">Manvi Gawande</a>
</div>
""", unsafe_allow_html=True)
