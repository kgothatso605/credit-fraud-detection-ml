"""
Azure ML Managed Online Endpoint - Scoring Script
Model: XGBoost (best model, ROC-AUC: 0.9518)

Accepts flat transaction JSON with customer and terminal profile fields.
Returns fraud probability and risk level per transaction.
"""
import json
import os
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

FEATURE_EXCLUDE = {
    "TX_FRAUD", "TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID"
}

CUSTOMER_COLS = [
    "CUSTOMER_ID", "x_customer_id", "y_customer_id",
    "mean_amount", "std_amount", "mean_nb_tx_per_day", "nb_terminals"
]

TERMINAL_COLS = ["TERMINAL_ID", "x_terminal_id", "y_terminal_id"]
TX_CORE_COLS  = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"]


def init():
    global model, feature_engineer
    model_dir = os.getenv("AZUREML_MODEL_DIR", os.path.join(os.path.dirname(__file__), "../../models"))
    model_dir = os.path.realpath(model_dir)

    model_path = os.path.join(model_dir, "xgboost.pkl")
    fe_path    = os.path.join(model_dir, "feature_engineer.pkl")

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading feature engineer from %s", fe_path)
    feature_engineer = joblib.load(fe_path)

    logger.info("Scoring script initialised — XGBoost fraud detector ready.")


def run(raw_data: str) -> str:
    try:
        payload = json.loads(raw_data)

        # Accept either {"transactions": [...]} or a single record dict
        records = payload.get("transactions", None)
        if records is None:
            records = [payload]

        transactions_df = pd.DataFrame(records)
        transactions_df["TX_DATETIME"] = pd.to_datetime(transactions_df["TX_DATETIME"])

        # Extract sub-tables expected by FeatureEngineer.transform()
        cust_cols_present = [c for c in CUSTOMER_COLS if c in transactions_df.columns]
        term_cols_present = [c for c in TERMINAL_COLS  if c in transactions_df.columns]

        customers  = transactions_df[cust_cols_present].drop_duplicates("CUSTOMER_ID").reset_index(drop=True)
        terminals  = transactions_df[term_cols_present].drop_duplicates("TERMINAL_ID").reset_index(drop=True)

        tx_cols = [c for c in TX_CORE_COLS if c in transactions_df.columns]
        tx_df   = transactions_df[tx_cols].copy()

        # Feature engineering (uses fitted terminal_stats from training)
        df = feature_engineer.transform(tx_df, customers, terminals)

        feature_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
        X = df[feature_cols].copy()
        X = X.select_dtypes(exclude=["object", "string"])
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        proba       = model.predict_proba(X)[:, 1]
        predictions = (proba >= 0.5).astype(int)

        results = []
        for i in range(len(transactions_df)):
            p = float(proba[i])
            results.append({
                "transaction_id":    str(transactions_df["TRANSACTION_ID"].iloc[i] if "TRANSACTION_ID" in transactions_df.columns else i),
                "fraud_probability": round(p, 6),
                "is_fraud":          int(predictions[i]),
                "risk_level":        "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.3 else "LOW",
                "amount":            float(transactions_df["TX_AMOUNT"].iloc[i]),
                "timestamp":         str(transactions_df["TX_DATETIME"].iloc[i]),
            })

        return json.dumps({
            "predictions": results,
            "model":       "xgboost",
            "scored_at":   datetime.now(timezone.utc).isoformat(),
            "count":       len(results),
        })

    except Exception as exc:
        logger.exception("Scoring error")
        return json.dumps({"error": str(exc)})
