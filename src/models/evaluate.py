"""
Model Evaluation - Phases 5-6
NO ACCURACY - only precision, recall, F1, PR-AUC
"""
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)


def evaluate_model(model, X, y, threshold=0.5):
    """Comprehensive model evaluation"""
    print("\n📊 Evaluating Model...")
    
    # Predict
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Metrics
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    pr_auc = average_precision_score(y, y_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'pr_auc': pr_auc,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
    }
    
    # Print results
    print(f"\n🎯 Results:")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1-Score: {f1:.2%}")
    print(f"   PR-AUC: {pr_auc:.3f}")
    
    return results


if __name__ == '__main__':
    import joblib
    
    model = joblib.load('models/fraud_detector.pkl')
    # Load validation data and evaluate
    print("✅ Evaluation module ready")
