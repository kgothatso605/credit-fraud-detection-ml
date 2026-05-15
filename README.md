# Credit Card Fraud Detection ML Framework

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Production-ready machine learning framework for credit card fraud detection with deployment capabilities

## 🎯 Project Overview

End-to-end fraud detection system built from scratch, achieving 76.28% precision with 9.04% recall through systematic methodology and professional ML practices.

**Key Achievement**: Terminal-based features account for 58% of model importance, demonstrating that *where* transactions occur matters more than *when* or *how much*.

### Dataset

- **Source**: Kaggle - Credit Card Fraud Detection Challenge
- **Size**: 291,231 training transactions, 226,731 test transactions
- **Imbalance**: 1:43 ratio (2.26% fraud rate)
- **Structure**: Multi-table relational data (transactions, customers, terminals)

**Acknowledgment**: We gratefully acknowledge Kaggle for providing this synthetic dataset for educational purposes.

## 📊 Results Summary

| Metric | Value | Status |
|--------|-------|--------|
| Precision | 76.28% | ✅ High |
| Recall | 9.04% | ⚠️  Conservative |
| F1-Score | 16.17% | Best balance |
| Training Time | 36.7s | ⚡ Fast |
| Top Feature | terminal_fraud_rate | 29.75% importance |

**Key Finding**: Discovered fundamental precision-recall constraint where 85%+ precision requires <1% recall.

## 🏗️ Architecture

```
End-to-End Pipeline:
Raw Data → Feature Engineering → Model Training → Evaluation → Deployment → API
```

### Features Engineered: 58 total

1. **Temporal (14)**: High-risk period (21:00-04:00), cyclical encodings
2. **Geographic (2)**: Distance calculations
3. **Amount Deviations (6)**: Personalized Z-scores per customer
4. **Terminal Risk (5)**: Historical fraud rates (58% of model importance!)
5. **Customer Behavioral (3)**: Usage patterns
6. **Advanced Contextual (7)**: Weekend patterns, customer time habits
7. **Interactions (6)**: Compound effects

### Models Evaluated: 20 configurations

- Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Techniques: Baseline, Class Weights, SMOTE, Threshold Tuning
- **Selected**: Random Forest Baseline (optimal precision-recall balance)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/credit_fraud_ml.git
cd credit_fraud_ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Usage

#### 1. Feature Engineering

```python
from src.features.feature_engineering import FeatureEngineer

# Initialize
engineer = FeatureEngineer()

# Load and engineer features
train_engineered = engineer.fit_transform(train_df, customers_df, terminals_df)
test_engineered = engineer.transform(test_df, customers_df, terminals_df)
```

#### 2. Model Training

```python
from src.models.train import train_model

# Train model
model, metrics = train_model(
    X_train, y_train,
    model_type='random_forest',
    save_path='models/fraud_detector.pkl'
)

print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
```

#### 3. Prediction

```python
from src.deployment.predictor import FraudPredictor

# Initialize predictor
predictor = FraudPredictor(model_path='models/fraud_detector.pkl')

# Predict
result = predictor.predict(transaction_data)
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
```

#### 4. API Deployment

```bash
# Start API server
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000

# Test prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"transaction_id": 12345, "amount": 100.50, ...}'
```

## 📁 Project Structure

```
credit_fraud_ml/
├── src/                      # Source code
│   ├── data/                # Data loading & validation
│   ├── features/            # Feature engineering
│   ├── models/              # Model training & evaluation
│   ├── deployment/          # API & serving
│   └── utils/               # Utilities
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
├── deployment/              # Docker & K8s configs
├── data/                    # Data (gitignored)
├── models/                  # Saved models (gitignored)
└── reports/                 # Figures & results
```

## 🔬 Methodology

### Phase 1: EDA (Days 1-3)

**Discoveries**:
- High-risk period: 21:00-04:00 (50% elevated fraud)
- Volume-risk paradox: Low-volume hours = higher fraud rates
- Amount analysis: Only $4.07 difference between fraud/legitimate
- Synthetic data recognition: Tuesday elevation (unusual pattern)

### Phase 2: Feature Engineering (Days 4-7)

**Created 58 features** across 7 categories with focus on:
- Terminal risk metrics (became 58% of model importance!)
- Personalized amount deviations (customer-specific baselines)
- Advanced temporal patterns (high-risk period indicators)

### Phase 3: Model Evaluation (Days 8-12)

**Systematic comparison**:
- Tested 20 configurations (5 models × 4 techniques)
- Discovered precision-recall constraint
- Random Forest baseline selected as optimal

### Phase 4: Deployment (Days 13-17)

**Production-ready**:
- FastAPI REST API
- Docker containerization
- Monitoring & logging
- Model versioning

## 🎓 Key Learnings

### 1. Feature Engineering > Model Selection
Terminal features dominate (58%) despite testing 5 different model types.

### 2. Data-Driven > Assumption-Driven  
Synthetic data showed Tuesday elevation (not weekends), teaching us to trust data over domain assumptions.

### 3. Systematic Evaluation Essential
Testing 20 configurations revealed constraints invisible from ad-hoc testing.

### 4. Honest Limitation Documentation
85% precision target proved impossible (requires <1% recall). Documenting why matters as much as successes.

## 📈 Performance Optimization

### Current Model
- **Precision**: 76.28%
- **Recall**: 9.04%
- **Inference Time**: <10ms per transaction

### Potential Improvements
1. Time-series features (rolling windows)
2. Network features (fraud ring detection)
3. Ensemble methods (stacking)
4. Deep learning (LSTM for sequences)

## 🐳 Deployment

### Docker

```bash
# Build image
docker build -t fraud-detector:latest .

# Run container
docker run -p 8000:8000 fraud-detector:latest
```

### Kubernetes

```bash
# Deploy to K8s
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods -l app=fraud-detector
```

### Monitoring

- Prometheus metrics at `/metrics`
- Health check at `/health`
- Model performance tracking

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_features.py -v
```

## 📝 API Documentation

Once running, visit: `http://localhost:8000/docs`

### Endpoints

**POST /predict** - Predict single transaction
```json
{
  "transaction_id": 12345,
  "amount": 100.50,
  "customer_id": 1,
  "terminal_id": 50,
  "timestamp": "2026-02-16T21:30:00"
}
```

**POST /predict/batch** - Batch predictions
**GET /health** - Health check
**GET /metrics** - Prometheus metrics

## 👤 Author

**Kgothatso Ntumbe**  
MSc Physics Student, University of the Witwatersrand  
Aspiring AI Engineer & Entrepreneur

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Dataset**: Kaggle - Credit Card Fraud Detection Challenge
- **Platform**: CERN SWAN for development
- **Tools**: scikit-learn, XGBoost, LightGBM, CatBoost, FastAPI

## 📚 Documentation

- [API Documentation](docs/api_documentation.md)
- [Feature Engineering Guide](docs/feature_engineering.md)
- [Deployment Guide](docs/deployment.md)

## 🗺️ Roadmap

- [x] EDA & Feature Engineering
- [x] Model Training & Evaluation
- [x] API Deployment
- [ ] Real-time Streaming Pipeline
- [ ] A/B Testing Framework
- [ ] AutoML Integration
- [ ] Model Monitoring Dashboard

---

**⭐ If you find this project useful, please consider giving it a star!**
