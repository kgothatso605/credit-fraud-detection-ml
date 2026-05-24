"""
Multi-Model Training Pipeline - Fraud Detection
Train:
- Random Forest
- XGBoost
- LightGBM
"""

import json
import time
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ==========================================================
# RANDOM FOREST
# ==========================================================

def train_random_forest(X_train, y_train):

    print("\n Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# ==========================================================
# XGBOOST
# ==========================================================

def train_xgboost(X_train, y_train):

    print("\n Training XGBoost...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# ==========================================================
# LIGHTGBM
# ==========================================================

def train_lightgbm(X_train, y_train):

    print("\n⚡ Training LightGBM...")

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# ==========================================================
# MAIN PIPELINE
# ==========================================================

if __name__ == '__main__':

    from src.data.data_loader import DataLoader
    from src.features.feature_engineering import FeatureEngineer

    print("\n Multi-Model Training Pipeline Started")
    print("=" * 60)

    # ======================================================
    # LOAD DATA
    # ======================================================

    loader = DataLoader()

    data = loader.load_all('train')

    # ======================================================
    # FEATURE ENGINEERING
    # ======================================================

    engineer = FeatureEngineer()

    df = engineer.fit_transform(**data)

    # Save feature engineer
    Path('models').mkdir(exist_ok=True)

    joblib.dump(
        engineer,
        'models/feature_engineer.pkl'
    )

    print("\n Saved feature engineer")

    # ======================================================
    # PREPARE FEATURES
    # ======================================================

    feature_cols = [c for c in df.columns if c not in
                    ['TX_FRAUD',
                     'TRANSACTION_ID',
                     'TX_DATETIME',
                     'CUSTOMER_ID',
                     'TERMINAL_ID']]

    X = df[feature_cols].copy()

    y = df['TX_FRAUD']

    print(f"\n Initial features: {X.shape[1]}")

    # ======================================================
    # CHECK FOR OBJECT COLUMNS
    # ======================================================

    print("\n🔍 Checking feature types...")

    object_cols = X.select_dtypes(
        include=['object', 'string']
    ).columns.tolist()

    if len(object_cols) > 0:

        print(f"\n Found {len(object_cols)} object columns:")

        for col in object_cols:

            print(f"\nColumn: {col}")

            sample_values = X[col].dropna().head(3).tolist()

            print("Sample values:")

            for val in sample_values:
                print(f"   {val}")

        print("\n Dropping object columns...")

        X = X.drop(columns=object_cols)

    else:
        print(" No object columns found")

    # ======================================================
    # CLEAN DATA
    # ======================================================

    # Convert to numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    # Replace infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing values
    X = X.fillna(0)

    print(f"\n Final features: {X.shape[1]}")
    print(" All features numeric and clean")

    # ======================================================
    # TRAIN / VALIDATION SPLIT
    # ======================================================

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"\n Training samples: {len(X_train):,}")
    print(f" Validation samples: {len(X_val):,}")

    # ======================================================
    # TRAIN MODELS
    # ======================================================

    models = {
        'random_forest': train_random_forest,
        'xgboost': train_xgboost,
        'lightgbm': train_lightgbm
    }

    trained_models = {}

    for model_name, trainer in models.items():

        print("\n" + "=" * 60)

        start = time.time()

        model = trainer(X_train, y_train)

        training_time = time.time() - start

        print(f"\n {model_name} training completed")
        print(f" Training time: {training_time:.1f} seconds")

        # Save model
        model_path = f'models/{model_name}.pkl'

        joblib.dump(model, model_path)

        print(f" Saved model: {model_path}")

        trained_models[model_name] = model

    # ======================================================
    # SAVE TRAINING METADATA
    # ======================================================

    metadata = {
        'models_trained': list(models.keys()),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'features': X.shape[1],
        'trained_at': datetime.now().isoformat()
    }

    with open('models/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(" ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)

    print("\n Saved Models:")

    for model_name in models.keys():
        print(f"   - models/{model_name}.pkl")

    print("\n Training pipeline complete!")