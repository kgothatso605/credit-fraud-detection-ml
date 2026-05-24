"""
Real-time transaction data simulator for the monitoring dashboard.
Streams transactions from test.csv / customers.csv / terminals.csv
and enriches them with customer+terminal profile fields so they
can be passed directly to FraudDetectorClient.score().
"""
import os
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT  = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR   = os.path.join(REPO_ROOT, "data", "raw")

_COLS_RENAME = {
    "x_customer_id": "x_customer_id",
    "y_customer_id": "y_customer_id",
}


class TransactionSimulator:
    """
    Loads historical test transactions and replays them one-by-one
    (or in small batches) to simulate a live transaction stream.

    Usage:
        sim = TransactionSimulator()
        sim.start(interval_seconds=1.0)
        ...
        batch = sim.get_batch(n=10)
        sim.stop()
    """

    MAX_BUFFER = 2_000

    def __init__(self):
        self._buffer: deque = deque(maxlen=self.MAX_BUFFER)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock    = threading.Lock()
        self._df      = self._load_data()
        self._idx     = 0

    # ── public ──────────────────────────────────────────────────────────────

    def start(self, interval_seconds: float = 1.0) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._produce,
            args=(interval_seconds,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def get_batch(self, n: int = 20) -> list[dict]:
        with self._lock:
            batch = []
            while self._buffer and len(batch) < n:
                batch.append(self._buffer.popleft())
            return batch

    def get_all_buffered(self) -> list[dict]:
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            return items

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    # ── internal ─────────────────────────────────────────────────────────────

    def _produce(self, interval: float) -> None:
        while self._running:
            record = self._next_record()
            with self._lock:
                self._buffer.append(record)
            time.sleep(interval)

    def _next_record(self) -> dict:
        row = self._df.iloc[self._idx % len(self._df)]
        self._idx += 1
        rec = row.to_dict()
        # Overwrite TX_DATETIME with *now* so the dashboard shows live timestamps
        rec["TX_DATETIME"] = datetime.now(timezone.utc).isoformat()
        return rec

    # ── data loading ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_data() -> pd.DataFrame:
        tx_path   = os.path.join(DATA_DIR, "test.csv")
        cust_path = os.path.join(DATA_DIR, "customers.csv")
        term_path = os.path.join(DATA_DIR, "terminals.csv")

        tx   = pd.read_csv(tx_path,   parse_dates=["TX_DATETIME"] if "TX_DATETIME" in pd.read_csv(tx_path, nrows=0).columns else [])
        cust = pd.read_csv(cust_path)
        term = pd.read_csv(term_path)

        # Normalise column names
        tx   = tx.rename(columns=str.upper)
        cust = cust.rename(columns=str.upper)
        term = term.rename(columns=str.upper)

        # Re-lowercase the profile columns expected by score.py
        cust_rename = {c: c.lower() for c in cust.columns if c != "CUSTOMER_ID"}
        term_rename = {c: c.lower() for c in term.columns if c != "TERMINAL_ID"}
        cust = cust.rename(columns=cust_rename)
        term = term.rename(columns=term_rename)

        df = tx.merge(cust, on="CUSTOMER_ID", how="left")
        df = df.merge(term, on="TERMINAL_ID", how="left")

        # Shuffle so the stream isn't chronologically ordered
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Ensure required columns exist with sensible defaults
        for col, default in [
            ("x_customer_id", 0.0), ("y_customer_id", 0.0),
            ("mean_amount",   100.0), ("std_amount",    30.0),
            ("mean_nb_tx_per_day", 2.0), ("nb_terminals", 3),
            ("x_terminal_id", 0.0), ("y_terminal_id", 0.0),
        ]:
            if col not in df.columns:
                df[col] = default

        return df
