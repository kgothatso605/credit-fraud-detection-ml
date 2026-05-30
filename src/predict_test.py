"""
Generate fraud predictions on the held-out test set.

Loads the fitted XGBoost model + FeatureEngineer, scores all 226,731
test transactions, and writes two output files:

  reports/test_predictions.csv   — full results with probabilities
  reports/submission.csv         — TRANSACTION_ID, TX_FRAUD (0/1)

Usage:
    python src/predict_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path("data/raw")
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

FEATURE_EXCLUDE = {
    "TX_FRAUD", "TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID"
}
THRESHOLD = 0.5   # default; set lower to improve recall

def load_data():
    print("Loading data ...")
    tx   = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["TX_DATETIME"])
    cust = pd.read_csv(DATA_DIR / "customers.csv")
    term = pd.read_csv(DATA_DIR / "terminals.csv")
    print(f"  Transactions : {len(tx):,}")
    print(f"  Customers    : {len(cust):,}")
    print(f"  Terminals    : {len(term):,}")
    return tx, cust, term

def engineer_features(tx, cust, term, fe):
    print("Engineering features ...")
    df = fe.transform(tx, cust, term)
    feat_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    X = df[feat_cols].copy()
    X = X.select_dtypes(exclude=["object", "string"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    print(f"  Feature matrix: {X.shape}")
    return df, X

def score(model, X, threshold=THRESHOLD):
    print(f"Scoring with XGBoost (threshold={threshold}) ...")
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    return proba, preds

def save_results(df, proba, preds):
    # Full results
    out = pd.DataFrame({
        "TRANSACTION_ID":    df["TRANSACTION_ID"].values,
        "TX_DATETIME":       df["TX_DATETIME"].values,
        "CUSTOMER_ID":       df["CUSTOMER_ID"].values,
        "TERMINAL_ID":       df["TERMINAL_ID"].values,
        "TX_AMOUNT":         df["TX_AMOUNT"].values,
        "fraud_probability": proba.round(6),
        "TX_FRAUD":          preds,
        "risk_level":        pd.Series(proba).apply(
            lambda p: "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.3 else "LOW"
        ).values,
    })
    full_path = REPORTS_DIR / "test_predictions.csv"
    out.to_csv(full_path, index=False)
    print(f"  Saved: {full_path}  ({len(out):,} rows)")

    # Submission file
    sub = out[["TRANSACTION_ID", "TX_FRAUD"]]
    sub_path = REPORTS_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")

    return out

def print_summary(out):
    n          = len(out)
    n_fraud    = out["TX_FRAUD"].sum()
    n_high     = (out["risk_level"] == "HIGH").sum()
    n_medium   = (out["risk_level"] == "MEDIUM").sum()
    n_low      = (out["risk_level"] == "LOW").sum()
    avg_prob   = out["fraud_probability"].mean()
    max_prob   = out["fraud_probability"].max()

    print("\n" + "=" * 55)
    print("TEST SET PREDICTIONS SUMMARY")
    print("=" * 55)
    print(f"  Total transactions  : {n:,}")
    print(f"  Flagged as fraud    : {n_fraud:,}  ({n_fraud/n*100:.2f}%)")
    print(f"  HIGH risk (≥0.70)   : {n_high:,}")
    print(f"  MEDIUM risk (≥0.30) : {n_medium:,}")
    print(f"  LOW risk (<0.30)    : {n_low:,}")
    print(f"  Avg fraud prob      : {avg_prob:.4f}")
    print(f"  Max fraud prob      : {max_prob:.4f}")
    print("=" * 55)

def main():
    print("\n=== XGBoost Test Set Prediction Pipeline ===\n")

    model = joblib.load(MODELS_DIR / "xgboost.pkl")
    fe    = joblib.load(MODELS_DIR / "feature_engineer.pkl")
    print("Models loaded.")

    tx, cust, term = load_data()
    df, X          = engineer_features(tx, cust, term, fe)
    proba, preds   = score(model, X)
    out            = save_results(df, proba, preds)
    print_summary(out)

if __name__ == "__main__":
    main()
