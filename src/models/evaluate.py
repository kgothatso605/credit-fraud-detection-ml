"""
Model Evaluation Pipeline - Fraud Detection

Evaluates:
- Random Forest
- XGBoost
- LightGBM

Metrics:
- Precision
- Recall
- F1-Score
- ROC-AUC

Generates:
- Confusion Matrix
- ROC Curve
- Feature Importance
- Model Comparison CSV
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(model_name, model, X_val, y_val):

    print("\n" + "=" * 60)
    print(f" Evaluating {model_name}")
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_val)

    y_proba = model.predict_proba(X_val)[:, 1]

    # Metrics
    precision = precision_score(y_val, y_pred)

    recall = recall_score(y_val, y_pred)

    f1 = f1_score(y_val, y_pred)

    roc_auc = roc_auc_score(y_val, y_proba)

    print(f"\n🎯 RESULTS ({model_name})")
    print(f"Precision : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print(f"F1-Score  : {f1:.2%}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\n Classification Report")
    print(classification_report(y_val, y_pred))

    return {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }, y_pred, y_proba


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def save_confusion_matrix(model_name, y_val, y_pred):

    cm = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(8, 6))

    plt.imshow(cm)

    plt.xticks([0, 1], ['Legitimate', 'Fraud'])

    plt.yticks([0, 1], ['Legitimate', 'Fraud'])

    plt.xlabel('Predicted')

    plt.ylabel('Actual')

    plt.title(f'{model_name} - Confusion Matrix')

    # Add text values
    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha='center',
                va='center'
            )

    plt.colorbar()

    plt.tight_layout()

    save_path = f'reports/figures/{model_name}_confusion_matrix.png'

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f" Saved: {save_path}")


# ==========================================================
# ROC CURVE
# ==========================================================

def save_roc_curve(model_name, y_val, y_proba):

    fpr, tpr, thresholds = roc_curve(y_val, y_proba)

    roc_auc = roc_auc_score(y_val, y_proba)

    plt.figure(figsize=(8, 6))

    plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(f'{model_name} ROC Curve (AUC={roc_auc:.4f})')

    plt.tight_layout()

    save_path = f'reports/figures/{model_name}_roc_curve.png'

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f" Saved: {save_path}")


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def save_feature_importance(model_name, model, X):

    if not hasattr(model, 'feature_importances_'):

        print(f" {model_name} has no feature_importances_")
        return

    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })

    importance = importance.sort_values(
        'importance',
        ascending=False
    )

    # Save CSV
    csv_path = f'reports/{model_name}_feature_importance.csv'

    importance.to_csv(csv_path, index=False)

    # Plot
    plt.figure(figsize=(12, 8))

    top_20 = importance.head(20)

    plt.barh(
        top_20['feature'],
        top_20['importance']
    )

    plt.gca().invert_yaxis()

    plt.xlabel('Importance')

    plt.title(f'{model_name} - Top 20 Features')

    plt.tight_layout()

    plot_path = f'reports/figures/{model_name}_feature_importance.png'

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f" Saved: {plot_path}")


# ==========================================================
# MAIN PIPELINE
# ==========================================================

if __name__ == '__main__':

    from src.data.data_loader import DataLoader
    from src.features.feature_engineering import FeatureEngineer

    print("\n MODEL EVALUATION PIPELINE")
    print("=" * 60)

    # ======================================================
    # CREATE DIRECTORIES
    # ======================================================

    Path('reports').mkdir(exist_ok=True)

    Path('reports/figures').mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # LOAD DATA
    # ======================================================

    loader = DataLoader()

    data = loader.load_all('train')

    engineer = FeatureEngineer()

    df = engineer.fit_transform(**data)

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

    # Remove object columns
    object_cols = X.select_dtypes(
        include=['object', 'string']
    ).columns.tolist()

    if len(object_cols) > 0:

        print("\n Dropping object columns:")
        print(object_cols)

        X = X.drop(columns=object_cols)

    # Clean data
    X = X.apply(pd.to_numeric, errors='coerce')

    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.fillna(0)

    # ======================================================
    # SPLIT
    # ======================================================

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ======================================================
    # LOAD MODELS
    # ======================================================

    models = {
        'random_forest':
            joblib.load('models/random_forest.pkl'),

        'xgboost':
            joblib.load('models/xgboost.pkl'),

        'lightgbm':
            joblib.load('models/lightgbm.pkl')
    }

    # ======================================================
    # EVALUATE ALL MODELS
    # ======================================================

    results = []

    for model_name, model in models.items():

        metrics, y_pred, y_proba = evaluate_model(
            model_name,
            model,
            X_val,
            y_val
        )

        results.append(metrics)

        # Save plots
        save_confusion_matrix(
            model_name,
            y_val,
            y_pred
        )

        save_roc_curve(
            model_name,
            y_val,
            y_proba
        )

        save_feature_importance(
            model_name,
            model,
            X
        )

    # ======================================================
    # SAVE COMPARISON RESULTS
    # ======================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        'roc_auc',
        ascending=False
    )

    results_df.to_csv(
        'reports/model_comparison.csv',
        index=False
    )

    with open('reports/model_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ======================================================
    # PRINT FINAL RESULTS
    # ======================================================

    print("\n" + "=" * 60)
    print("🏆 FINAL MODEL COMPARISON")
    print("=" * 60)

    print(results_df)

    print("\n Saved:")
    print("   - reports/model_comparison.csv")
    print("   - reports/model_results.json")
    print("   - reports/figures/*.png")

    print("\n Evaluation complete!")