#!/usr/bin/env python
"""
Generate and save all EDA + RL report figures.

Run from project root:
    python reports/generate_figures.py

Output structure:
    reports/figures/
        01_data_overview/
        02_univariate_analysis/
        03_bivariate_analysis/
        04_time_series_patterns/
        05_geographic_analysis/
        06_fraud_behavior_analysis/
        07_feature_correlations/
        08_model_readiness/       ← figures already written by evaluate.py
        09_rl_feature_exploration/
"""

import os
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.append(str(ROOT))

from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer

FIG_ROOT = ROOT / "reports" / "figures"
DPI = 150


def save(fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")
    print(f"  ✓  {fig_dir.name}/{name}.png")


# ── Load raw data (shared across all sections) ────────────────────────────────
print("Loading data …")
loader = DataLoader()
data = loader.load_all("train")
txn = data["transactions"].copy()
cust = data["customers"].copy()
term = data["terminals"].copy()

# Pre-compute time columns once
txn = txn.sort_values("TX_DATETIME").reset_index(drop=True)
txn["TX_DATE"] = txn["TX_DATETIME"].dt.date
txn["TX_HOUR"] = txn["TX_DATETIME"].dt.hour
txn["TX_DAY"]  = txn["TX_DATETIME"].dt.dayofweek
txn["TX_WEEK"] = txn["TX_DATETIME"].dt.isocalendar().week.astype(int)

# Full merge (reused by several sections)
merged = txn.merge(cust, on="CUSTOMER_ID", how="left").merge(
    term, on="TERMINAL_ID", how="left"
)
merged["distance"] = np.sqrt(
    (merged["x_customer_id"] - merged["x_terminal_id"]) ** 2
    + (merged["y_customer_id"] - merged["y_terminal_id"]) ** 2
)
merged["amount_vs_mean"] = merged["TX_AMOUNT"] - merged["mean_amount"]
merged["amount_ratio"]   = merged["TX_AMOUNT"] / (merged["mean_amount"] + 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 01 — Data Overview
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[01] Data Overview")
d01 = FIG_ROOT / "01_data_overview"

try:
    import missingno as msno
    msno.matrix(txn)
    plt.title("Missing Values Matrix – Transactions")
    plt.tight_layout()
    save(d01, "01_missing_values_matrix")
except ImportError:
    print("  ! missingno not installed – skipping missing values matrix")

fraud_counts = txn["TX_FRAUD"].value_counts()
plt.figure(figsize=(6, 6))
fraud_counts.plot(
    kind="pie",
    autopct="%1.2f%%",
    labels=["Legit", "Fraud"],
    colors=["steelblue", "crimson"],
)
plt.title("Fraud Distribution")
plt.ylabel("")
save(d01, "02_fraud_distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# 02 — Univariate Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[02] Univariate Analysis")
d02 = FIG_ROOT / "02_univariate_analysis"

plot_cols = ["TX_AMOUNT", "TX_FRAUD", "CUSTOMER_ID", "TERMINAL_ID"]

for col in plot_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(txn[col], bins=50, color="steelblue")
    plt.title(f"{col} – Distribution")
    plt.tight_layout()
    save(d02, f"hist_{col.lower()}")

for col in plot_cols:
    plt.figure(figsize=(10, 3))
    sns.boxplot(x=txn[col], color="darkorange")
    plt.title(f"{col} – Boxplot")
    plt.tight_layout()
    save(d02, f"box_{col.lower()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 03 — Bivariate Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[03] Bivariate Analysis")
d03 = FIG_ROOT / "03_bivariate_analysis"

# TX_AMOUNT vs fraud
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x="TX_FRAUD", y="TX_AMOUNT", data=txn, ax=axes[0])
axes[0].set_title("TX_AMOUNT by Fraud Label")
for label, grp in txn.groupby("TX_FRAUD"):
    sns.kdeplot(
        grp["TX_AMOUNT"], ax=axes[1],
        label="Fraud" if label else "Legit", fill=True, alpha=0.4,
    )
axes[1].set_title("TX_AMOUNT KDE: Fraud vs Legit")
axes[1].legend()
plt.tight_layout()
save(d03, "01_tx_amount_vs_fraud")

# Fraud rate by hour & day
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
hourly = txn.groupby("TX_HOUR")["TX_FRAUD"].mean()
axes[0].bar(hourly.index, hourly.values, color="steelblue")
axes[0].set_title("Fraud Rate by Hour of Day")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("Fraud Rate")
daily = txn.groupby("TX_DAY")["TX_FRAUD"].mean()
daily.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
axes[1].bar(daily.index, daily.values, color="darkorange")
axes[1].set_title("Fraud Rate by Day of Week")
axes[1].set_ylabel("Fraud Rate")
plt.tight_layout()
save(d03, "02_fraud_rate_by_time")

# Fraud rate by amount bin
txn_tmp = txn.copy()
txn_tmp["amount_bin"] = pd.cut(txn_tmp["TX_AMOUNT"], bins=10)
amount_fraud = txn_tmp.groupby("amount_bin", observed=True)["TX_FRAUD"].mean()
plt.figure(figsize=(12, 4))
plt.bar(range(len(amount_fraud)), amount_fraud.values, color="mediumseagreen")
plt.xticks(
    range(len(amount_fraud)),
    [str(b) for b in amount_fraud.index],
    rotation=45, ha="right",
)
plt.title("Fraud Rate by Transaction Amount Bin")
plt.ylabel("Fraud Rate")
plt.tight_layout()
save(d03, "03_fraud_rate_by_amount_bin")

# Customer features vs fraud
customer_features = ["mean_amount", "std_amount", "mean_nb_tx_per_day", "nb_terminals"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, feat in zip(axes.flatten(), customer_features):
    sns.boxplot(x="TX_FRAUD", y=feat, data=merged, ax=ax)
    ax.set_title(f"{feat} vs TX_FRAUD")
plt.suptitle("Customer Features vs Fraud", fontsize=13)
plt.tight_layout()
save(d03, "04_customer_features_vs_fraud")

# Top 20 terminals by fraud rate
term_stats = txn.groupby("TERMINAL_ID").agg(
    fraud_rate=("TX_FRAUD", "mean"), tx_count=("TX_FRAUD", "count")
).reset_index()
top_terms = term_stats[term_stats["tx_count"] >= 20].nlargest(20, "fraud_rate")
plt.figure(figsize=(12, 5))
plt.bar(range(len(top_terms)), top_terms["fraud_rate"].values, color="crimson")
plt.xticks(range(len(top_terms)), top_terms["TERMINAL_ID"].astype(str), rotation=45)
plt.title("Top 20 Terminals by Fraud Rate (min 20 transactions)")
plt.ylabel("Fraud Rate")
plt.tight_layout()
save(d03, "05_top_terminals_by_fraud_rate")

# Bivariate correlation heatmap (raw txn columns)
drop_ids = ["TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"]
corr_raw = txn.drop(columns=drop_ids, errors="ignore").corr(numeric_only=True)
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_raw, dtype=bool))
sns.heatmap(
    corr_raw, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, square=True,
)
plt.title("Transaction Feature Correlation Matrix")
plt.tight_layout()
save(d03, "06_correlation_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# 04 — Time Series Patterns
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[04] Time Series Patterns")
d04 = FIG_ROOT / "04_time_series_patterns"

daily = txn.groupby("TX_DATE").agg(
    total=("TX_FRAUD", "count"), fraud=("TX_FRAUD", "sum")
)
daily["fraud_rate"] = daily["fraud"] / daily["total"]

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
axes[0].plot(daily.index, daily["total"], color="steelblue")
axes[0].set_title("Daily Transaction Volume")
axes[0].set_ylabel("Count")
axes[1].plot(daily.index, daily["fraud"], color="crimson")
axes[1].set_title("Daily Fraud Count")
axes[1].set_ylabel("Fraud Transactions")
axes[2].plot(daily.index, daily["fraud_rate"], color="darkorange")
axes[2].set_title("Daily Fraud Rate")
axes[2].set_ylabel("Fraud Rate")
axes[2].set_xlabel("Date")
plt.tight_layout()
save(d04, "01_daily_volume_and_fraud_rate")

txn["rolling_fraud_rate"] = txn["TX_FRAUD"].rolling(1000).mean()
plt.figure(figsize=(16, 5))
plt.plot(txn["TX_DATETIME"], txn["rolling_fraud_rate"], color="purple", linewidth=0.8)
plt.title("Rolling Fraud Rate (1 000-transaction window)")
plt.xlabel("Date")
plt.ylabel("Fraud Rate")
plt.tight_layout()
save(d04, "02_rolling_fraud_rate")

pivot = txn.groupby(["TX_DAY", "TX_HOUR"])["TX_FRAUD"].mean().unstack()
pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
plt.figure(figsize=(16, 5))
sns.heatmap(pivot, cmap="YlOrRd", annot=False, linewidths=0.3)
plt.title("Fraud Rate Heatmap: Hour of Day × Day of Week")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.tight_layout()
save(d04, "03_hourly_fraud_heatmap")

weekly = txn.groupby("TX_WEEK").agg(
    total=("TX_FRAUD", "count"), fraud=("TX_FRAUD", "sum")
)
weekly["fraud_rate"] = weekly["fraud"] / weekly["total"]
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].bar(weekly.index, weekly["fraud"], color="crimson", alpha=0.8)
axes[0].set_title("Weekly Fraud Count")
axes[0].set_ylabel("Fraud Transactions")
axes[1].plot(weekly.index, weekly["fraud_rate"], marker="o", color="darkorange")
axes[1].set_title("Weekly Fraud Rate")
axes[1].set_ylabel("Fraud Rate")
axes[1].set_xlabel("Week Number")
plt.tight_layout()
save(d04, "04_weekly_fraud_trend")

fraud_only = txn[txn["TX_FRAUD"] == 1]
daily_fraud_amt = fraud_only.groupby("TX_DATE")["TX_AMOUNT"].mean()
plt.figure(figsize=(16, 5))
plt.plot(daily_fraud_amt.index, daily_fraud_amt.values, color="firebrick", linewidth=1.2)
plt.title("Average Fraudulent Transaction Amount Over Time")
plt.xlabel("Date")
plt.ylabel("Mean TX_AMOUNT")
plt.tight_layout()
save(d04, "05_avg_fraud_amount_over_time")


# ═══════════════════════════════════════════════════════════════════════════════
# 05 — Geographic Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[05] Geographic Analysis")
d05 = FIG_ROOT / "05_geographic_analysis"

fraud_txns = merged[merged["TX_FRAUD"] == 1]
legit_txns = merged[merged["TX_FRAUD"] == 0]
term_geo = (
    merged.groupby(["TERMINAL_ID", "x_terminal_id", "y_terminal_id"])
    .agg(fraud_rate=("TX_FRAUD", "mean"), tx_count=("TX_FRAUD", "count"))
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].scatter(cust["x_customer_id"], cust["y_customer_id"],
                alpha=0.4, s=15, color="steelblue")
axes[0].set_title("Customer Locations")
axes[1].scatter(term["x_terminal_id"], term["y_terminal_id"],
                alpha=0.6, s=20, color="darkorange", marker="^")
axes[1].set_title("Terminal Locations")
plt.suptitle("Spatial Distribution of Customers and Terminals", fontsize=13)
plt.tight_layout()
save(d05, "01_customer_terminal_locations")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].scatter(legit_txns["x_terminal_id"], legit_txns["y_terminal_id"],
                alpha=0.1, s=5, color="steelblue")
axes[0].set_title("Legitimate Transaction Terminals")
axes[1].scatter(fraud_txns["x_terminal_id"], fraud_txns["y_terminal_id"],
                alpha=0.3, s=10, color="crimson")
axes[1].set_title("Fraudulent Transaction Terminals")
plt.suptitle("Terminal Locations: Legit vs Fraud", fontsize=13)
plt.tight_layout()
save(d05, "02_legit_vs_fraud_hotspots")

active = term_geo[term_geo["tx_count"] >= 20].copy()
plt.figure(figsize=(12, 9))
sc = plt.scatter(
    active["x_terminal_id"], active["y_terminal_id"],
    c=active["fraud_rate"], cmap="YlOrRd",
    s=active["tx_count"] / active["tx_count"].max() * 300 + 20,
    alpha=0.8, edgecolors="grey", linewidths=0.3,
)
plt.colorbar(sc, label="Fraud Rate")
plt.title("Terminal Fraud Rate Map\n(colour = fraud rate, size = transaction volume)")
plt.tight_layout()
save(d05, "03_terminal_fraud_rate_map")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x="TX_FRAUD", y="distance", data=merged, ax=axes[0])
axes[0].set_title("Customer–Terminal Distance by Fraud Label")
for label, grp in merged.groupby("TX_FRAUD"):
    sns.kdeplot(grp["distance"], ax=axes[1],
                label="Fraud" if label else "Legit", fill=True, alpha=0.4)
axes[1].set_title("Distance Distribution: Fraud vs Legit")
axes[1].legend()
plt.tight_layout()
save(d05, "04_customer_terminal_distance")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].set_title("Fraud Density (Customer locations)")
sns.kdeplot(x=fraud_txns["x_customer_id"], y=fraud_txns["y_customer_id"],
            cmap="Reds", fill=True, thresh=0.05, ax=axes[0])
axes[0].scatter(fraud_txns["x_customer_id"], fraud_txns["y_customer_id"],
                s=2, alpha=0.1, color="darkred")
axes[1].set_title("Fraud Density (Terminal locations)")
sns.kdeplot(x=fraud_txns["x_terminal_id"], y=fraud_txns["y_terminal_id"],
            cmap="Oranges", fill=True, thresh=0.05, ax=axes[1])
axes[1].scatter(fraud_txns["x_terminal_id"], fraud_txns["y_terminal_id"],
                s=2, alpha=0.1, color="darkorange")
plt.suptitle("2D Fraud Density Maps", fontsize=13)
plt.tight_layout()
save(d05, "05_2d_fraud_density")


# ═══════════════════════════════════════════════════════════════════════════════
# 06 — Fraud Behavior Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[06] Fraud Behavior Analysis")
d06 = FIG_ROOT / "06_fraud_behavior_analysis"

customer_tx = txn.groupby("CUSTOMER_ID").agg(
    tx_count=("TX_FRAUD", "count"),
    fraud_count=("TX_FRAUD", "sum"),
    fraud_rate=("TX_FRAUD", "mean"),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(customer_tx["tx_count"], bins=40, ax=axes[0], color="steelblue")
axes[0].set_title("Transactions per Customer")
sns.histplot(customer_tx["fraud_count"], bins=40, ax=axes[1], color="crimson")
axes[1].set_title("Fraud Count per Customer")
sns.histplot(customer_tx["fraud_rate"], bins=40, ax=axes[2], color="darkorange")
axes[2].set_title("Fraud Rate per Customer")
plt.tight_layout()
save(d06, "01_customer_tx_frequency")

term_stats2 = txn.groupby("TERMINAL_ID").agg(
    tx_count=("TX_FRAUD", "count"),
    fraud_count=("TX_FRAUD", "sum"),
    fraud_rate=("TX_FRAUD", "mean"),
).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(term_stats2["fraud_rate"], bins=40, ax=axes[0], color="purple")
axes[0].set_title("Fraud Rate per Terminal")
sns.scatterplot(data=term_stats2, x="tx_count", y="fraud_rate",
                alpha=0.5, ax=axes[1], color="mediumseagreen")
axes[1].set_title("Terminal Volume vs Fraud Rate")
plt.tight_layout()
save(d06, "02_terminal_fraud_rate_distribution")

repeat_victims = customer_tx[customer_tx["fraud_count"] > 0].copy()
repeat_victims["victim_type"] = repeat_victims["fraud_count"].apply(
    lambda x: "1 fraud" if x == 1 else "2 frauds" if x == 2 else "3+ frauds"
)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
vc = repeat_victims["victim_type"].value_counts()
axes[0].bar(vc.index, vc.values, color=["#f4a460", "#e07b54", "#b94040"])
axes[0].set_title("Fraud Victims by Number of Incidents")
sns.boxplot(x="victim_type", y="tx_count", data=repeat_victims,
            order=["1 fraud", "2 frauds", "3+ frauds"], ax=axes[1])
axes[1].set_title("Transaction Volume of Fraud Victims")
plt.tight_layout()
save(d06, "03_repeat_fraud_victims")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x="TX_FRAUD", y="amount_vs_mean", data=merged, ax=axes[0])
axes[0].set_title("TX_AMOUNT − Customer Mean Amount")
axes[0].axhline(0, color="grey", linestyle="--", linewidth=0.8)
for label, grp in merged.groupby("TX_FRAUD"):
    sns.kdeplot(grp["amount_ratio"].clip(upper=10), ax=axes[1],
                label="Fraud" if label else "Legit", fill=True, alpha=0.4)
axes[1].set_title("Amount Ratio vs Customer Mean (clipped at 10×)")
axes[1].legend()
plt.tight_layout()
save(d06, "04_fraud_amount_vs_customer_profile")

txn_sorted = txn.sort_values(["CUSTOMER_ID", "TX_DATETIME"])
txn_sorted["prev_tx_time"] = txn_sorted.groupby("CUSTOMER_ID")["TX_DATETIME"].shift(1)
txn_sorted["time_gap_min"] = (
    txn_sorted["TX_DATETIME"] - txn_sorted["prev_tx_time"]
).dt.total_seconds() / 60
gap_data = txn_sorted.dropna(subset=["time_gap_min"])
gap_data = gap_data[gap_data["time_gap_min"] <= 1440]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x="TX_FRAUD", y="time_gap_min", data=gap_data, ax=axes[0])
axes[0].set_title("Time Gap to Previous Transaction vs Fraud")
for label, grp in gap_data.groupby("TX_FRAUD"):
    sns.kdeplot(grp["time_gap_min"], ax=axes[1],
                label="Fraud" if label else "Legit", fill=True, alpha=0.4)
axes[1].set_title("Time Gap Distribution (≤24 h)")
axes[1].legend()
plt.tight_layout()
save(d06, "05_time_gap_between_transactions")

cust_profile = customer_tx.merge(cust, on="CUSTOMER_ID", how="left")
high_risk = cust_profile[cust_profile["fraud_rate"] >= 0.1]
low_risk  = cust_profile[cust_profile["fraud_rate"] <  0.1]
profile_feats = ["mean_amount", "std_amount", "mean_nb_tx_per_day", "nb_terminals"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, feat in zip(axes.flatten(), profile_feats):
    ax.hist(low_risk[feat].dropna(),  bins=30, alpha=0.6, label="Low risk",
            color="steelblue", density=True)
    ax.hist(high_risk[feat].dropna(), bins=30, alpha=0.6, label="High risk",
            color="crimson",   density=True)
    ax.set_title(feat)
    ax.legend()
plt.suptitle("Customer Profile: High-Risk (≥10% fraud) vs Low-Risk", fontsize=13)
plt.tight_layout()
save(d06, "06_high_vs_low_risk_customer_profiles")


# ═══════════════════════════════════════════════════════════════════════════════
# 07 — Feature Correlations
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[07] Feature Correlations")
d07 = FIG_ROOT / "07_feature_correlations"

df7 = merged.copy()
df7["TX_WEEK"] = df7["TX_DATETIME"].dt.isocalendar().week.astype(int)
drop_ids = ["TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"]
corr7 = df7.drop(columns=drop_ids, errors="ignore").corr(numeric_only=True)

plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr7, dtype=bool))
sns.heatmap(
    corr7, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, square=True,
    linewidths=0.4, annot_kws={"size": 7},
)
plt.title("Feature Correlation Matrix (lower triangle)", fontsize=13)
plt.tight_layout()
save(d07, "01_full_correlation_heatmap")

fraud_corr = corr7["TX_FRAUD"].drop("TX_FRAUD").sort_values()
colors = ["crimson" if v > 0 else "steelblue" for v in fraud_corr]
plt.figure(figsize=(10, 7))
fraud_corr.plot(kind="barh", color=colors, edgecolor="black", linewidth=0.4)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with TX_FRAUD")
plt.xlabel("Pearson Correlation")
plt.tight_layout()
save(d07, "02_fraud_correlation_ranked_bar")

top6 = fraud_corr.abs().sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
rng = np.random.default_rng(42)
for ax, feat in zip(axes.flatten(), top6):
    for label, grp in df7.groupby("TX_FRAUD"):
        ax.scatter(
            grp[feat],
            grp["TX_FRAUD"] + rng.uniform(-0.1, 0.1, len(grp)),
            alpha=0.05, s=3,
            label="Fraud" if label else "Legit",
        )
    ax.set_xlabel(feat)
    ax.set_title(f"{feat}  (r={corr7['TX_FRAUD'][feat]:.3f})")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Legit", "Fraud"])
plt.suptitle("Top 6 Features vs TX_FRAUD (jitter plot)", fontsize=13)
plt.tight_layout()
save(d07, "03_top6_features_jitter_plot")

corr_pairs = (
    corr7.where(np.tril(np.ones(corr7.shape), k=-1).astype(bool))
    .stack()
    .reset_index()
)
corr_pairs.columns = ["Feature A", "Feature B", "Correlation"]
corr_pairs["abs_corr"] = corr_pairs["Correlation"].abs()
high_corr = corr_pairs[corr_pairs["abs_corr"] > 0.3].sort_values("abs_corr", ascending=False)
if len(high_corr) > 0:
    plt.figure(figsize=(10, max(4, len(high_corr) * 0.4)))
    hc_colors = ["crimson" if v > 0 else "steelblue" for v in high_corr["Correlation"]]
    plt.barh(
        high_corr["Feature A"] + "  ×  " + high_corr["Feature B"],
        high_corr["Correlation"], color=hc_colors, edgecolor="black", linewidth=0.4,
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Highly Correlated Feature Pairs (|r| > 0.3)")
    plt.xlabel("Pearson Correlation")
    plt.tight_layout()
    save(d07, "04_high_correlation_pairs")

top4 = fraud_corr.abs().sort_values(ascending=False).head(4).index.tolist()
sample = df7[top4 + ["TX_FRAUD"]].sample(n=3000, random_state=42).copy()
sample["TX_FRAUD"] = sample["TX_FRAUD"].map({0: "Legit", 1: "Fraud"})
g = sns.pairplot(
    sample, hue="TX_FRAUD",
    palette={"Legit": "steelblue", "Fraud": "crimson"},
    plot_kws={"alpha": 0.3, "s": 10}, diag_kind="kde",
)
g.figure.suptitle("Pairplot of Top 4 Features by Fraud Correlation", y=1.01, fontsize=13)
plt.tight_layout()
save(d07, "05_pairplot_top4_features")


# ═══════════════════════════════════════════════════════════════════════════════
# 08 — Model Readiness (figures already written by evaluate.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[08] Model Readiness — figures already in reports/figures/ (written by evaluate.py)")


# ═══════════════════════════════════════════════════════════════════════════════
# 09 — RL Feature Exploration (quick run: 15 ep × 8 steps)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[09] RL Feature Exploration (quick run)")
d09 = FIG_ROOT / "09_rl_feature_exploration"

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    fe = FeatureEngineer()
    df_feat = fe.fit_transform(
        data["transactions"], data["customers"], data["terminals"]
    )
    drop_c = ["TX_FRAUD", "TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID", "TX_DATETIME"]
    X = df_feat.drop(columns=[c for c in drop_c if c in df_feat.columns], errors="ignore")
    X = X.select_dtypes(include=np.number).fillna(0)
    y = df_feat["TX_FRAUD"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    all_feats = list(X.columns)

    def _evaluate(feats):
        if not feats:
            return 0.5
        m = lgb.LGBMClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=3, num_leaves=10,
            class_weight="balanced", random_state=42, verbosity=-1,
        )
        m.fit(X_tr[feats], y_tr)
        return roc_auc_score(y_val, m.predict_proba(X_val[feats])[:, 1])

    class _Env:
        def __init__(self):
            self.selected = set()
        def reset(self):
            k = random.randint(1, min(5, len(all_feats)))
            self.selected = set(random.sample(all_feats, k))
            return frozenset(self.selected)
        def step(self, action):
            op, f = action
            if op == "add" and f not in self.selected:
                self.selected.add(f)
            elif op == "remove" and len(self.selected) > 1:
                self.selected.discard(f)
            auc = _evaluate(list(self.selected))
            return frozenset(self.selected), auc - 0.5, auc
        def actions(self):
            return (
                [("add", f) for f in all_feats if f not in self.selected]
                + [("remove", f) for f in all_feats if f in self.selected and len(self.selected) > 1]
            )

    Q = defaultdict(float)
    epsilon = 0.5

    def _act(state, avail):
        if random.random() < epsilon:
            return random.choice(avail)
        return max(avail, key=lambda a: Q[(state, a[0], a[1])])

    def _update(s, a, r, ns, na, alpha=0.2, gamma=0.9):
        key = (s, a[0], a[1])
        nq = max((Q[(ns, x[0], x[1])] for x in na), default=0.0)
        Q[key] += alpha * (r + gamma * nq - Q[key])

    N_EP, N_STEPS = 15, 8
    random.seed(42)
    env9 = _Env()
    ep_aucs9, visits9, best_auc9, best_feats9 = [], defaultdict(int), 0.0, []

    print(f"  Running {N_EP} episodes × {N_STEPS} steps …")
    for ep in range(N_EP):
        state = env9.reset()
        ep_auc = 0.5
        for _ in range(N_STEPS):
            avail = env9.actions()
            if not avail:
                break
            action = _act(state, avail)
            ns, reward, auc = env9.step(action)
            _update(state, action, reward, ns, env9.actions())
            state = ns
            ep_auc = auc
            for f in env9.selected:
                visits9[f] += 1
            if auc > best_auc9:
                best_auc9 = auc
                best_feats9 = list(env9.selected)
        epsilon = max(0.05, epsilon * 0.97)
        ep_aucs9.append(ep_auc)

    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rolling9 = pd.Series(ep_aucs9).rolling(3, min_periods=1).mean()
    axes[0].plot(range(1, N_EP + 1), ep_aucs9, alpha=0.4, color="steelblue", label="Episode AUC")
    axes[0].plot(range(1, N_EP + 1), rolling9, color="steelblue", linewidth=2, label="3-ep rolling")
    axes[0].axhline(best_auc9, color="crimson", linestyle="--", label=f"Best {best_auc9:.4f}")
    axes[0].set_title("AUC per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(range(1, N_EP + 1), pd.Series(ep_aucs9).cummax(), color="darkorange", linewidth=2)
    axes[1].set_title("Cumulative Best AUC")
    axes[1].set_xlabel("Episode")
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("RL Feature Selection – Learning Curves", fontsize=13)
    plt.tight_layout()
    save(d09, "01_learning_curves")

    # Feature visit frequency
    if visits9:
        vs = sorted(visits9.items(), key=lambda x: x[1], reverse=True)[:30]
        fn9, vc9 = zip(*vs)
        plt.figure(figsize=(12, 8))
        c9 = ["crimson" if f in best_feats9 else "steelblue" for f in fn9]
        plt.barh(list(fn9), list(vc9), color=c9, edgecolor="black", linewidth=0.4)
        plt.gca().invert_yaxis()
        plt.title("Top 30 Feature Visit Frequency\n(red = in best subset)")
        plt.xlabel("Times Visited")
        plt.tight_layout()
        save(d09, "02_feature_visit_frequency")

    # RL-selected vs all
    auc_all = _evaluate(all_feats)
    auc_rl  = _evaluate(best_feats9)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [f"All Features\n(n={len(all_feats)})", f"RL-Selected\n(n={len(best_feats9)})"],
        [auc_all, auc_rl],
        color=["steelblue", "crimson"], edgecolor="black", width=0.45,
    )
    for bar, val in zip(bars, [auc_all, auc_rl]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", fontweight="bold")
    ax.set_ylim(max(0.5, min(auc_all, auc_rl) - 0.02), 1.01)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("All Features vs RL-Selected Subset")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save(d09, "03_rl_vs_all_features_comparison")

    # RL-selected feature importances
    if best_feats9:
        fm = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=31,
            class_weight="balanced", random_state=42, verbosity=-1,
        )
        fm.fit(X_tr[best_feats9], y_tr)
        imp_df = pd.DataFrame(
            {"feature": best_feats9, "importance": fm.feature_importances_}
        ).sort_values("importance", ascending=True)
        plt.figure(figsize=(10, max(4, len(best_feats9) * 0.4)))
        plt.barh(imp_df["feature"], imp_df["importance"],
                 color="teal", edgecolor="black", linewidth=0.4)
        plt.title("Feature Importances – RL-Selected Subset (LightGBM)")
        plt.xlabel("Importance")
        plt.tight_layout()
        save(d09, "04_rl_selected_feature_importances")

except ImportError as e:
    print(f"  ! Skipping section 09 – missing dependency: {e}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All figures saved under reports/figures/")
all_pngs = sorted(FIG_ROOT.rglob("*.png"))
for p in all_pngs:
    rel = p.relative_to(FIG_ROOT)
    print(f"  {rel}")
print(f"\nTotal: {len(all_pngs)} figures")
