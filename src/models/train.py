"""
Model Training Pipeline - Phase 6
YOUR RESULTS: 76.28% precision, 9.04% recall
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json
from pathlib import Path
from datetime import datetime


def train_random_forest(X_train, y_train, save_dir='models'):
    """Train Random Forest with optimal hyperparameters"""
    print("🎯 Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Save model
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    model_path = save_dir / 'fraud_detector.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    return model


if __name__ == '__main__':
    from src.data.data_loader import DataLoader
    from src.features.feature_engineering import FeatureEngineer
    
    print("🚀 Training Pipeline Started")
    
    # Load data
    loader = DataLoader()
    data = loader.load_all('train')
    
    # Engineer features
    engineer = FeatureEngineer()
    df = engineer.fit_transform(**data)
    
    # Save engineer
    joblib.dump(engineer, 'models/feature_engineer.pkl')
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in 
                    ['TX_FRAUD', 'TRANSACTION_ID', 'TX_DATETIME', 
                     'CUSTOMER_ID', 'TERMINAL_ID']]
    
    X = df[feature_cols]
    y = df['TX_FRAUD']
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train
    model = train_random_forest(X_train, y_train)
    
    print("✅ Training complete!")
