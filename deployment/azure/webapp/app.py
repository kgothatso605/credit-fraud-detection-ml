"""
Credit Fraud Detection — FastAPI Scoring API
Deployed to Azure App Service (no ACR required).

Endpoints:
  GET  /          → health check
  POST /score     → score one or many transactions
  GET  /model/info → model metadata
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── path setup ───────────────────────────────────────────────────────────────
# When deployed: MODELS_DIR env var points to the models/ directory.
# Locally: resolve relative to this file.
_HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.getenv("MODELS_DIR", os.path.join(_HERE, "..", "..", "..", "models"))
MODELS_DIR  = os.path.realpath(MODELS_DIR)

# Allow importing src/ when running locally
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Fraud Detection API",
    description="Real-time fraud scoring — XGBoost (ROC-AUC 0.9518)",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── model loading ─────────────────────────────────────────────────────────────
_model = None
_fe    = None


def _load_models():
    global _model, _fe
    if _model is None:
        logger.info("Loading models from %s", MODELS_DIR)
        _model = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
        _fe    = joblib.load(os.path.join(MODELS_DIR, "feature_engineer.pkl"))
        logger.info("Models loaded OK")


FEATURE_EXCLUDE = {
    "TX_FRAUD", "TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID"
}
CUSTOMER_COLS = [
    "CUSTOMER_ID", "x_customer_id", "y_customer_id",
    "mean_amount", "std_amount", "mean_nb_tx_per_day", "nb_terminals"
]
TERMINAL_COLS = ["TERMINAL_ID", "x_terminal_id", "y_terminal_id"]
TX_CORE_COLS  = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"]


# ── schemas ───────────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    TRANSACTION_ID:      int    = Field(0)
    TX_DATETIME:         str    = Field(..., example="2026-05-24 21:30:00")
    CUSTOMER_ID:         int    = Field(..., example=1)
    TERMINAL_ID:         int    = Field(..., example=5)
    TX_AMOUNT:           float  = Field(..., example=350.0)
    x_customer_id:       float  = Field(25.0)
    y_customer_id:       float  = Field(40.0)
    mean_amount:         float  = Field(80.0)
    std_amount:          float  = Field(30.0)
    mean_nb_tx_per_day:  float  = Field(2.0)
    nb_terminals:        int    = Field(4)
    x_terminal_id:       float  = Field(50.0)
    y_terminal_id:       float  = Field(60.0)


class ScoreRequest(BaseModel):
    transactions: list[Transaction]


class Prediction(BaseModel):
    transaction_id:    str
    fraud_probability: float
    is_fraud:          int
    risk_level:        str
    amount:            float
    timestamp:         str
    latency_ms:        int


class ScoreResponse(BaseModel):
    predictions: list[Prediction]
    model:       str
    scored_at:   str
    count:       int


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    _load_models()


@app.get("/")
def health():
    return {
        "status":  "healthy",
        "model":   "xgboost",
        "roc_auc": 0.9518,
        "time":    datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/info")
def model_info():
    return {
        "model":          "xgboost",
        "roc_auc":        0.9518,
        "precision":      0.821,
        "recall":         0.070,
        "f1_score":       0.129,
        "training_size":  232984,
        "features":       51,
        "trained_at":     "2026-05-15T13:42:44",
        "models_dir":     MODELS_DIR,
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    _load_models()

    try:
        records = [t.model_dump() for t in request.transactions]
        df      = pd.DataFrame(records)
        df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

        cust_cols = [c for c in CUSTOMER_COLS if c in df.columns]
        term_cols = [c for c in TERMINAL_COLS  if c in df.columns]
        tx_cols   = [c for c in TX_CORE_COLS   if c in df.columns]

        customers = df[cust_cols].drop_duplicates("CUSTOMER_ID").reset_index(drop=True)
        terminals = df[term_cols].drop_duplicates("TERMINAL_ID").reset_index(drop=True)
        tx_df     = df[tx_cols].copy()

        engineered = _fe.transform(tx_df, customers, terminals)

        feat_cols = [c for c in engineered.columns if c not in FEATURE_EXCLUDE]
        X = engineered[feat_cols].copy()
        X = X.select_dtypes(exclude=["object", "string"])
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        t0    = time.perf_counter()
        proba = _model.predict_proba(X)[:, 1]
        ms    = int((time.perf_counter() - t0) * 1000)

        preds = []
        for i in range(len(df)):
            p = float(proba[i])
            preds.append(Prediction(
                transaction_id    = str(df["TRANSACTION_ID"].iloc[i]),
                fraud_probability = round(p, 6),
                is_fraud          = int(p >= 0.5),
                risk_level        = "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.3 else "LOW",
                amount            = float(df["TX_AMOUNT"].iloc[i]),
                timestamp         = str(df["TX_DATETIME"].iloc[i]),
                latency_ms        = ms,
            ))

        return ScoreResponse(
            predictions = preds,
            model       = "xgboost",
            scored_at   = datetime.now(timezone.utc).isoformat(),
            count       = len(preds),
        )

    except Exception as exc:
        logger.exception("Scoring error")
        raise HTTPException(status_code=500, detail=str(exc))
