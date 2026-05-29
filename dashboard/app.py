"""
Credit Fraud Detection — Real-Time Monitoring Dashboard
Powered by XGBoost on Azure ML (ROC-AUC 0.9518)

Run:
    cd dashboard
    streamlit run app.py
"""
import os
import sys
import time
from datetime import datetime, timezone
from collections import deque

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Load .env so AZURE_ENDPOINT_URL / AZURE_ENDPOINT_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard.utils.azure_client import FraudDetectorClient
from dashboard.utils.data_simulator import TransactionSimulator

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #1e2130;
    border-radius: 10px;
    padding: 12px 18px;
    border-left: 4px solid #4c78a8;
}
.high-risk  { color: #ff4b4b; font-weight: bold; }
.medium-risk{ color: #ffa500; font-weight: bold; }
.low-risk   { color: #21c354; font-weight: bold; }
.alert-card {
    background: #2d1f1f;
    border-left: 4px solid #ff4b4b;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "scored_transactions": deque(maxlen=500),
        "fraud_ts":            [],   # (timestamp, fraud_rate)
        "total_scored":        0,
        "total_fraud":         0,
        "latencies":           deque(maxlen=100),
        "simulator":           None,
        "client":              None,
        "running":             False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Fraud Monitor")
    st.markdown("---")

    st.subheader("Azure ML Endpoint")
    endpoint_url = st.text_input(
        "Scoring URI",
        value=os.getenv("AZURE_ENDPOINT_URL", ""),
        placeholder="https://<endpoint>.inference.ml.azure.com/score",
        type="default",
    )
    api_key = st.text_input(
        "API Key",
        value=os.getenv("AZURE_ENDPOINT_KEY", ""),
        placeholder="Leave blank to use local model",
        type="password",
    )

    st.markdown("---")
    st.subheader("Stream Settings")
    refresh_rate   = st.slider("Refresh interval (s)", 1, 10, 2)
    batch_size     = st.slider("Transactions per batch", 1, 20, 5)
    fraud_threshold= st.slider("Fraud threshold", 0.1, 0.9, 0.5, 0.05)

    st.markdown("---")
    col_start, col_stop = st.columns(2)
    start_btn = col_start.button("▶ Start", use_container_width=True, type="primary")
    stop_btn  = col_stop.button("⏹ Stop",  use_container_width=True)

    st.markdown("---")
    st.subheader("Model Info")
    st.markdown("""
| | |
|---|---|
| **Algorithm** | XGBoost |
| **ROC-AUC** | 0.9518 |
| **Precision** | 0.821 |
| **Recall** | 0.070 |
| **Train size** | 232,984 |
| **Features** | 51 |
""")

    st.markdown("---")
    if st.button("Reset stats", use_container_width=True):
        st.session_state.scored_transactions = deque(maxlen=500)
        st.session_state.fraud_ts    = []
        st.session_state.total_scored = 0
        st.session_state.total_fraud  = 0
        st.session_state.latencies    = deque(maxlen=100)
        st.rerun()


# ── Start / stop logic ────────────────────────────────────────────────────────
if start_btn and not st.session_state.running:
    st.session_state.client    = FraudDetectorClient(endpoint_url, api_key)
    st.session_state.simulator = TransactionSimulator()
    st.session_state.simulator.start(interval_seconds=0.3)
    st.session_state.running   = True

if stop_btn and st.session_state.running:
    if st.session_state.simulator:
        st.session_state.simulator.stop()
    st.session_state.running = False


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Credit Fraud Detection — Live Monitor")
mode_badge = (
    "🟢 Azure ML Endpoint" if (endpoint_url and api_key)
    else "🟡 Local Model (no endpoint configured)"
)
st.caption(f"{mode_badge}  •  Workspace: AL_ML_LEARNING_COURSE  •  Model: XGBoost")
st.markdown("---")


# ── Score a batch if running ───────────────────────────────────────────────────
if st.session_state.running and st.session_state.simulator:
    raw_batch = st.session_state.simulator.get_batch(n=batch_size)
    if raw_batch:
        try:
            predictions = st.session_state.client.score(raw_batch)
            for pred in predictions:
                pred["fraud_threshold"] = fraud_threshold
                pred["is_fraud_thresh"] = int(pred["fraud_probability"] >= fraud_threshold)
                st.session_state.scored_transactions.appendleft(pred)
                if pred["is_fraud_thresh"]:
                    st.session_state.total_fraud += 1
                st.session_state.total_scored += 1
                if "latency_ms" in pred:
                    st.session_state.latencies.append(pred["latency_ms"])

            # Record fraud rate time series point
            n = st.session_state.total_scored
            rate = (st.session_state.total_fraud / n * 100) if n > 0 else 0.0
            st.session_state.fraud_ts.append(
                (datetime.now(timezone.utc).isoformat(), rate)
            )
        except Exception as exc:
            st.error(f"Scoring error: {exc}")


# ── KPI row ───────────────────────────────────────────────────────────────────
total    = st.session_state.total_scored
fraud_n  = st.session_state.total_fraud
legit_n  = total - fraud_n
rate_pct = (fraud_n / total * 100) if total > 0 else 0.0
avg_lat  = int(np.mean(list(st.session_state.latencies))) if st.session_state.latencies else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Scored",   f"{total:,}")
kpi2.metric("Legitimate",     f"{legit_n:,}")
kpi3.metric("Fraud Detected", f"{fraud_n:,}",
            delta=f"+{fraud_n}" if fraud_n > 0 else None,
            delta_color="inverse")
kpi4.metric("Fraud Rate",     f"{rate_pct:.2f}%")
kpi5.metric("Avg Latency",    f"{avg_lat} ms")

st.markdown("---")


# ── Charts row ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Fraud Rate Over Time")
    if st.session_state.fraud_ts:
        ts_df = pd.DataFrame(st.session_state.fraud_ts[-120:], columns=["time", "fraud_rate"])
        ts_df["time"] = pd.to_datetime(ts_df["time"])
        fig = px.line(
            ts_df, x="time", y="fraud_rate",
            labels={"fraud_rate": "Fraud Rate (%)", "time": "Time"},
            color_discrete_sequence=["#ff4b4b"],
        )
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,33,48,0.5)",
            font_color="#cdd6f4",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        fig.add_hline(y=2.0, line_dash="dash", line_color="#ffa500",
                      annotation_text="Alert threshold (2%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start the stream to see fraud rate trend.")

with col_right:
    st.subheader("Risk Distribution")
    scored = list(st.session_state.scored_transactions)
    if scored:
        risk_counts = pd.Series([t["risk_level"] for t in scored]).value_counts()
        colors = {"HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "LOW": "#21c354"}
        fig_pie = go.Figure(go.Pie(
            labels=risk_counts.index.tolist(),
            values=risk_counts.values.tolist(),
            marker_colors=[colors.get(l, "#888") for l in risk_counts.index],
            hole=0.45,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cdd6f4",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Waiting for data...")

st.markdown("---")


# ── Fraud probability histogram + Feature importance ─────────────────────────
col_hist, col_feat = st.columns([1, 1])

with col_hist:
    st.subheader("Fraud Probability Distribution")
    scored = list(st.session_state.scored_transactions)
    if scored:
        probs = [t["fraud_probability"] for t in scored]
        fig_hist = px.histogram(
            x=probs, nbins=40,
            labels={"x": "Fraud Probability", "y": "Count"},
            color_discrete_sequence=["#4c78a8"],
        )
        fig_hist.add_vline(
            x=fraud_threshold, line_dash="dash",
            line_color="#ff4b4b", annotation_text=f"Threshold ({fraud_threshold})"
        )
        fig_hist.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,33,48,0.5)",
            font_color="#cdd6f4",
            bargap=0.05,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Waiting for data...")

with col_feat:
    st.subheader("Top Feature Importances (XGBoost)")
    feat_path = os.path.join(
        os.path.dirname(__file__), "../reports/xgboost_feature_importance.csv"
    )
    if os.path.exists(feat_path):
        feat_df = pd.read_csv(feat_path).head(15)
        fig_feat = px.bar(
            feat_df.sort_values("importance"),
            x="importance", y="feature", orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            labels={"importance": "Importance", "feature": ""},
        )
        fig_feat.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,33,48,0.5)",
            font_color="#cdd6f4",
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Feature importance file not found.")

st.markdown("---")


# ── High-risk alerts + Live transaction feed ──────────────────────────────────
col_alerts, col_feed = st.columns([1, 2])

with col_alerts:
    st.subheader("🚨 High-Risk Alerts")
    scored = list(st.session_state.scored_transactions)
    high   = [t for t in scored if t["risk_level"] == "HIGH"][:10]
    if high:
        for tx in high:
            prob = tx["fraud_probability"]
            amt  = tx.get("amount", 0)
            tid  = tx.get("transaction_id", "N/A")
            ts   = str(tx.get("timestamp", ""))[:19]
            st.markdown(
                f'<div class="alert-card">'
                f'<b>TX {tid}</b><br>'
                f'Amount: <b>R{amt:,.2f}</b> &nbsp;|&nbsp; '
                f'Prob: <b class="high-risk">{prob:.1%}</b><br>'
                f'<span style="color:#888;font-size:0.75rem">{ts}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("No high-risk transactions detected yet.")

with col_feed:
    st.subheader("Live Transaction Feed")
    scored = list(st.session_state.scored_transactions)[:50]
    if scored:
        feed_df = pd.DataFrame([{
            "Timestamp":  str(t.get("timestamp", ""))[:19],
            "TX ID":      t.get("transaction_id", ""),
            "Amount":     t.get("amount", 0),
            "Fraud Prob": t["fraud_probability"],
            "Risk":       t["risk_level"],
            "Flagged":    "YES" if t["is_fraud_thresh"] else "no",
        } for t in scored])

        def colour_row(row):
            if row["Risk"] == "HIGH":
                return ["background-color: rgba(255,75,75,0.15)"] * len(row)
            if row["Risk"] == "MEDIUM":
                return ["background-color: rgba(255,165,0,0.10)"] * len(row)
            return [""] * len(row)

        styled = (
            feed_df.style
            .apply(colour_row, axis=1)
            .format({"Amount": "R{:,.2f}", "Fraud Prob": "{:.4f}"})
        )
        st.dataframe(styled, use_container_width=True, height=320)
    else:
        st.info("No transactions scored yet. Click ▶ Start to begin.")

st.markdown("---")


# ── Amount vs Fraud Probability scatter ──────────────────────────────────────
st.subheader("Amount vs Fraud Probability")
scored = list(st.session_state.scored_transactions)
if len(scored) >= 5:
    scatter_df = pd.DataFrame([{
        "Amount":        t.get("amount", 0),
        "Fraud Prob":    t["fraud_probability"],
        "Risk":          t["risk_level"],
    } for t in scored])
    colour_map = {"HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "LOW": "#21c354"}
    fig_sc = px.scatter(
        scatter_df, x="Amount", y="Fraud Prob", color="Risk",
        color_discrete_map=colour_map,
        opacity=0.6,
        labels={"Fraud Prob": "Fraud Probability"},
    )
    fig_sc.add_hline(y=fraud_threshold, line_dash="dash", line_color="#888",
                     annotation_text="Threshold")
    fig_sc.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,33,48,0.5)",
        font_color="#cdd6f4",
        legend_title_text="Risk Level",
    )
    st.plotly_chart(fig_sc, use_container_width=True)
else:
    st.info("Score at least 5 transactions to see the scatter plot.")


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()
