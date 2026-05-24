"""
Azure ML endpoint client for the monitoring dashboard.
Falls back to local model inference when no endpoint is configured.
"""
import json
import logging
import os
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

REPO_ROOT  = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(REPO_ROOT, "models")

FEATURE_EXCLUDE = {
    "TX_FRAUD", "TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID"
}
CUSTOMER_COLS = [
    "CUSTOMER_ID", "x_customer_id", "y_customer_id",
    "mean_amount", "std_amount", "mean_nb_tx_per_day", "nb_terminals"
]
TERMINAL_COLS = ["TERMINAL_ID", "x_terminal_id", "y_terminal_id"]
TX_CORE_COLS  = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"]


class FraudDetectorClient:
    """
    Scores transactions against the fraud model.

    Priority:
      1. Azure ML Managed Online Endpoint  (if url + key configured)
      2. Local model fallback              (models/ directory)
    """

    def __init__(self, endpoint_url: str = "", api_key: str = ""):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key      = api_key
        self._local_model = None
        self._local_fe    = None

    # ── public ──────────────────────────────────────────────────────────────

    def score(self, transactions: list[dict]) -> list[dict]:
        """Score a list of transaction dicts; returns prediction dicts."""
        if self.endpoint_url and self.api_key:
            return self._score_remote(transactions)
        return self._score_local(transactions)

    def health_check(self) -> dict:
        """Return dict with keys: mode, latency_ms, ok."""
        if self.endpoint_url and self.api_key:
            return self._remote_health()
        return {"mode": "local", "latency_ms": 0, "ok": True}

    # ── remote ──────────────────────────────────────────────────────────────

    def _score_remote(self, transactions: list[dict]) -> list[dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({"transactions": transactions})
        t0 = time.perf_counter()
        resp = requests.post(self.endpoint_url, data=payload, headers=headers, timeout=30)
        latency = int((time.perf_counter() - t0) * 1000)
        resp.raise_for_status()
        result = resp.json()
        preds  = result.get("predictions", [])
        for p in preds:
            p["latency_ms"] = latency
            p["source"] = "azure"
        return preds

    def _remote_health(self) -> dict:
        try:
            t0   = time.perf_counter()
            resp = requests.get(self.endpoint_url, timeout=5)
            ms   = int((time.perf_counter() - t0) * 1000)
            return {"mode": "azure", "latency_ms": ms, "ok": resp.status_code < 500}
        except Exception as exc:
            return {"mode": "azure", "latency_ms": -1, "ok": False, "error": str(exc)}

    # ── local ────────────────────────────────────────────────────────────────

    def _load_local(self) -> None:
        if self._local_model is None:
            self._local_model = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
            self._local_fe    = joblib.load(os.path.join(MODELS_DIR, "feature_engineer.pkl"))

    def _score_local(self, transactions: list[dict]) -> list[dict]:
        self._load_local()

        df = pd.DataFrame(transactions)
        df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

        cust_cols = [c for c in CUSTOMER_COLS if c in df.columns]
        term_cols = [c for c in TERMINAL_COLS  if c in df.columns]
        tx_cols   = [c for c in TX_CORE_COLS   if c in df.columns]

        customers = df[cust_cols].drop_duplicates("CUSTOMER_ID").reset_index(drop=True)
        terminals = df[term_cols].drop_duplicates("TERMINAL_ID").reset_index(drop=True)
        tx_df     = df[tx_cols].copy()

        engineered = self._local_fe.transform(tx_df, customers, terminals)

        feat_cols = [c for c in engineered.columns if c not in FEATURE_EXCLUDE]
        X = engineered[feat_cols].copy()
        X = X.select_dtypes(exclude=["object", "string"])
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        t0    = time.perf_counter()
        proba = self._local_model.predict_proba(X)[:, 1]
        ms    = int((time.perf_counter() - t0) * 1000)

        results = []
        for i in range(len(df)):
            p = float(proba[i])
            results.append({
                "transaction_id":    str(df["TRANSACTION_ID"].iloc[i] if "TRANSACTION_ID" in df.columns else i),
                "fraud_probability": round(p, 6),
                "is_fraud":          int(p >= 0.5),
                "risk_level":        "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.3 else "LOW",
                "amount":            float(df["TX_AMOUNT"].iloc[i]),
                "timestamp":         str(df["TX_DATETIME"].iloc[i]),
                "latency_ms":        ms,
                "source":            "local",
            })
        return results
