# Credit Card Fraud Detection Using Ensemble Machine Learning and Real-Time Deployment on Azure

**Author:** Kgothatso Ntumbe  
**Student Number:** 2445026  
**Institution:** University of the Witwatersrand (Wits), Johannesburg  
**Course:** AI/ML Learning Course  
**Date:** May 2026  
**Repository:** https://github.com/kgothatso605/credit-fraud-detection-ml  
**Live API:** https://fraud-detect-api-2445026.azurewebsites.net

---

## Abstract

This paper presents an end-to-end machine learning pipeline for real-time credit card fraud detection. Using a synthetic multi-table relational dataset of 291,231 training transactions, we engineer 58 features across seven semantic categories and evaluate three gradient-boosted ensemble models — XGBoost, LightGBM, and Random Forest. XGBoost achieves the highest discriminative performance with a ROC-AUC of **0.9518** and a precision of **0.821**, selected as the production model. The pipeline is deployed as a live REST API on Microsoft Azure App Service, with a real-time Streamlit monitoring dashboard that streams transaction scoring results. A key finding is that **terminal-level features account for 58% of total model importance**, establishing that *where* a transaction occurs is the strongest fraud signal — more predictive than transaction amount or time of day.

---

## 1. Introduction

Credit card fraud costs the global financial industry over $32 billion annually (Nilson Report, 2023). Traditional rule-based detection systems suffer from high false-positive rates and cannot adapt to evolving fraud patterns. Machine learning offers a data-driven alternative capable of learning complex, non-linear fraud signatures from historical transaction data.

The core challenges in fraud detection are:

1. **Severe class imbalance**: Fraud accounts for roughly 2% of transactions, making naive classifiers useless.
2. **Precision-recall trade-off**: High precision (few false alarms) conflicts with high recall (catching all fraud).
3. **Real-time requirements**: Decisions must be made in milliseconds at transaction time.
4. **Feature engineering**: Raw transaction data must be transformed into meaningful behavioural signals.

This work addresses all four challenges through systematic feature engineering, model selection with appropriate metrics, threshold analysis, and cloud deployment.

---

## 2. Dataset

### 2.1 Source

The dataset is a synthetic credit card fraud simulation sourced from Kaggle, designed to mimic real-world transaction patterns while preserving privacy.

### 2.2 Structure

The dataset consists of three relational tables:

| Table | Rows | Key Columns |
|---|---|---|
| `train_transactions.csv` | 291,231 | TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT, TX_FRAUD |
| `test.csv` | 226,731 | TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT |
| `customers.csv` | 1,000 | CUSTOMER_ID, x_customer_id, y_customer_id, mean_amount, std_amount, mean_nb_tx_per_day, nb_terminals |
| `terminals.csv` | 2,000 | TERMINAL_ID, x_terminal_id, y_terminal_id |

### 2.3 Class Imbalance

| Class | Count | Percentage |
|---|---|---|
| Legitimate (0) | 284,634 | 97.74% |
| Fraud (1) | 6,597 | 2.26% |
| **Imbalance Ratio** | **1:43** | |

The severe imbalance means a naive model that always predicts "legitimate" achieves 97.74% accuracy — a misleading metric. We therefore focus on **ROC-AUC** and **precision-recall** metrics throughout.

### 2.4 Train/Validation Split

The training data is split 80/20 with stratification to preserve the fraud rate:

| Split | Transactions | Fraud Cases |
|---|---|---|
| Train | 232,984 | ~5,278 |
| Validation | 58,247 | ~1,319 |

---

## 3. Exploratory Data Analysis

### 3.1 Key Findings from EDA (Notebooks 01–09)

**Temporal Patterns (Notebook 04)**
- Fraud peaks sharply between **21:00 and 04:00**, with 21:00 being the single highest-risk hour.
- The "clubbing hypothesis": late-night weekend transactions show disproportionate fraud.
- Weekday nights (21:00–04:00) show elevated fraud compared to weekend nights of the same hours.

**Geographic Patterns (Notebook 05)**
- Fraudulent transactions cluster at terminals far from a customer's typical location.
- Customer-terminal Euclidean distance is a strong fraud signal.
- Terminal hotspots are geographically concentrated, suggesting compromised devices.

**Behavioral Patterns (Notebook 06)**
- Repeat fraud victims exist: some customers appear in multiple fraud transactions.
- High-frequency customers (many transactions/day) show different fraud profiles than occasional users.
- Fraud amounts tend to be larger than the customer's personal mean, often >2 standard deviations above their norm.

**Feature Correlations (Notebook 07)**
- `terminal_fraud_rate` has the highest point-biserial correlation with `TX_FRAUD`.
- `terminal_fraud_count` and `terminal_fraud_rate` are highly correlated with each other (expected — they measure the same phenomenon at different scales).
- Amount deviation features (Z-scores) correlate moderately with fraud.

**RL Feature Exploration (Notebook 09)**
- A reinforcement learning agent was used to explore the feature space.
- The RL-selected feature subset consistently included `terminal_fraud_rate`, `is_high_risk_period`, and `amount_deviation` as the top three features, independently confirming the supervised learning importance rankings.

---

## 4. Feature Engineering

58 features were engineered across seven categories:

### 4.1 Temporal Features (14)

| Feature | Description |
|---|---|
| `tx_hour`, `tx_day_of_week`, `tx_day`, `tx_month` | Raw time components |
| `is_high_risk_period` | 1 if hour ∈ {21, 22, 23, 0, 1, 2, 3, 4} |
| `is_peak_fraud_hour` | 1 if hour == 21 |
| `is_hour_23` | 1 if hour == 23 |
| `is_weekend` | 1 if day_of_week ∈ {5, 6} |
| `is_friday` | 1 if day_of_week == 4 |
| `is_business_hours` | 1 if hour ∈ [9, 17] |
| `hour_sin`, `hour_cos` | Cyclical hour encoding |
| `day_sin`, `day_cos` | Cyclical day-of-week encoding |

### 4.2 Geographic Features (2)

| Feature | Formula |
|---|---|
| `distance_to_terminal` | √((x_cust − x_term)² + (y_cust − y_term)²) |
| `manhattan_distance` | |x_cust − x_term| + |y_cust − y_term| |

### 4.3 Amount Deviation Features (6)

Personalised Z-scores per customer using their historical mean and standard deviation:

| Feature | Description |
|---|---|
| `amount_deviation` | (TX_AMOUNT − mean_amount) / (std_amount + ε) |
| `amount_ratio` | TX_AMOUNT / (mean_amount + ε) |
| `is_high_amount` | 1 if amount_deviation > 2 |
| `is_low_amount` | 1 if amount_deviation < −2 |
| `amount_log` | log(TX_AMOUNT + 1) |

### 4.4 Terminal Risk Features (5) — 58% of Model Importance

These are computed from training data only and carried forward to test/inference:

| Feature | Training-time computation |
|---|---|
| `terminal_fraud_rate` | fraud_count / tx_count per terminal |
| `terminal_fraud_count` | total frauds at this terminal |
| `terminal_tx_count` | total transactions at this terminal |
| `terminal_avg_amount` | mean transaction amount at this terminal |
| `terminal_std_amount` | std of transaction amount at this terminal |

New terminals at inference time are assigned the overall training fraud rate (1.9%).

### 4.5 Customer Behavioral Features (3)

| Feature | Description |
|---|---|
| `customer_terminal_diversity` | Number of distinct terminals used (nb_terminals) |
| `is_frequent_customer` | 1 if mean_nb_tx_per_day > median |
| `customer_spending_consistency` | mean_amount / (std_amount + ε) |

### 4.6 Advanced Contextual Features (7)

Weekend/night interaction patterns including `is_clubbing_time`, `weekday_night`, `weekend_amount_deviation`.

### 4.7 Interaction Features (6)

Compound effects: `distance_unusual_time`, `high_amount_weekday_night`, `risky_hour_terminal`, `triple_unusual`, `weekend_unusual_amount`, `night_unusual_amount`.

---

## 5. Model Development

### 5.1 Models Evaluated

Three gradient-boosted ensemble models were trained with balanced class weights to address the 1:43 imbalance:

- **XGBoost** (XGBClassifier, scale_pos_weight=43)
- **LightGBM** (LGBMClassifier, class_weight='balanced')
- **Random Forest** (RandomForestClassifier, class_weight='balanced')

### 5.2 Validation Results

| Model | ROC-AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **XGBoost** ✓ | **0.9518** | **0.821** | 0.070 | 0.129 |
| LightGBM | 0.9425 | 0.832 | 0.068 | 0.125 |
| Random Forest | 0.9071 | 0.158 | 0.710 | 0.259 |

### 5.3 Model Selection: XGBoost

XGBoost was selected as the production model for the following reasons:

1. **Highest ROC-AUC (0.9518)**: ROC-AUC measures the model's ability to rank fraudulent transactions above legitimate ones across all possible thresholds. It is the most informative metric for imbalanced classification because it is threshold-independent.

2. **High precision (0.821)**: When XGBoost flags a transaction as fraud, it is correct 82.1% of the time. This minimises false alarms, which erode customer trust.

3. **Threshold flexibility**: With a default threshold of 0.5, recall is low (7%). The model outputs calibrated probabilities, so the threshold can be lowered to any desired operating point without retraining.

### 5.4 The Precision-Recall Trade-Off

The low recall (7%) at threshold=0.5 is a deliberate operational decision, not a model failure. The model's ROC-AUC of 0.9518 means it has strong discriminative power — it is simply set to a conservative operating point.

| Threshold | Precision | Recall | Fraud Caught |
|---|---|---|---|
| 0.50 (deployed) | 0.821 | 0.070 | ~92 / 1,319 |
| 0.30 | ~0.60 | ~0.25 | ~330 / 1,319 |
| 0.10 | ~0.35 | ~0.55 | ~725 / 1,319 |

For a production system, the threshold is a business decision: high-value fraud warrants a lower threshold at the cost of more false positives.

### 5.5 Feature Importance (XGBoost)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `terminal_fraud_rate` | 29.75% |
| 2 | `terminal_fraud_count` | 28.29% |
| 3 | `amount_deviation` | 8.4% |
| 4 | `is_high_risk_period` | 6.2% |
| 5 | `distance_to_terminal` | 4.1% |
| **Top 2** | **Terminal features** | **58.04%** |

**Key insight**: The history of fraud at a specific terminal is by far the strongest predictor. A new transaction at a historically fraudulent terminal is highly likely to be fraud regardless of amount or time.

---

## 6. Test Set Predictions

The trained XGBoost model was applied to all 226,731 held-out test transactions:

| Metric | Value |
|---|---|
| Total test transactions | 226,731 |
| Flagged as fraud (threshold=0.5) | 567 (0.25%) |
| HIGH risk (probability ≥ 0.70) | 77 |
| MEDIUM risk (probability 0.30–0.70) | 3,091 |
| LOW risk (probability < 0.30) | 223,563 |
| Mean fraud probability | 0.0217 |
| Maximum fraud probability | 0.9878 |

Output files:
- `reports/test_predictions.csv` — full predictions with probabilities and risk levels
- `reports/submission.csv` — TRANSACTION_ID, TX_FRAUD format

---

## 7. Deployment Architecture

### 7.1 Overview

```
Transaction Request (JSON)
        │
        ▼
┌─────────────────────────────────┐
│  Azure App Service              │
│  (South Africa North region)    │
│                                 │
│  FastAPI Application            │
│  ├── Feature Engineering        │
│  │   (FeatureEngineer.pkl)      │
│  └── XGBoost Scoring            │
│      (xgboost.pkl)              │
└─────────────────────────────────┘
        │
        ▼
  JSON Response
  {fraud_probability, risk_level}
        │
        ▼
┌─────────────────────────────────┐
│  Streamlit Dashboard            │
│  (Local / Cloud)                │
│  ├── Live KPI metrics           │
│  ├── Fraud rate trend chart     │
│  ├── Risk distribution pie      │
│  ├── High-risk alert cards      │
│  └── Transaction feed table     │
└─────────────────────────────────┘
```

### 7.2 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — returns model name and ROC-AUC |
| `/score` | POST | Score one or more transactions |
| `/model/info` | GET | Full model metadata |
| `/docs` | GET | Interactive Swagger UI |

**Live API:** https://fraud-detect-api-2445026.azurewebsites.net

### 7.3 Deployment Details

| Component | Technology |
|---|---|
| Cloud provider | Microsoft Azure |
| Service | Azure App Service (Linux, Python 3.11) |
| Region | South Africa North |
| SKU | B1 (Basic) |
| Framework | FastAPI + Uvicorn |
| Average latency | ~96ms per batch |
| Authentication | None (open endpoint for demonstration) |

### 7.4 Deployment Challenges and Solutions

| Challenge | Solution |
|---|---|
| Azure ML ACR not found in resource group | Switched from Azure ML Managed Online Endpoint to Azure App Service |
| Azure for Students policy blocking 21 regions | Probed all 102 Azure regions; South Africa North was allowed |
| Azure CLI pyexpat crash on macOS/Python 3.13 | Treated post-creation CLI crash as non-fatal (webapp was created successfully before crash) |

---

## 8. Monitoring Dashboard

The Streamlit dashboard provides real-time visibility into the deployed model's scoring activity:

**Dashboard panels:**
1. **KPI Row** — Total scored, fraud count, fraud rate %, average latency
2. **Fraud Rate Over Time** — Line chart with 2% alert threshold line
3. **Risk Distribution** — Pie chart (HIGH / MEDIUM / LOW)
4. **Fraud Probability Histogram** — Distribution of scores with adjustable threshold
5. **Feature Importance** — Top 15 XGBoost features (bar chart)
6. **High-Risk Alert Cards** — Live feed of HIGH-risk transactions
7. **Transaction Feed** — Colour-coded scrolling table
8. **Amount vs Probability Scatter** — Relationship between transaction size and fraud risk

The dashboard streams transactions from the historical test set to simulate a live feed, calling the Azure endpoint for real-time scoring.

---

## 9. Key Findings

1. **Terminal history dominates fraud signals (58% importance)**: The single most powerful fraud indicator is the historical fraud rate of the terminal where a transaction occurs. Terminals that have processed fraud before are far more likely to process fraud again. This suggests physical device compromise is a primary fraud vector.

2. **Time of day matters, but less than terminal history**: The 21:00–04:00 window is high-risk, but only the 6th most important feature. Temporal features alone are insufficient for reliable fraud detection.

3. **Personalised amount deviations outperform raw amounts**: A transaction of R850 means different things for a customer who usually spends R50 vs. one who usually spends R1,000. Z-score normalisation per customer is necessary.

4. **Precision-recall cannot be simultaneously maximised at this class ratio**: With a 1:43 imbalance, any classifier faces a fundamental trade-off. XGBoost's ROC-AUC of 0.9518 indicates excellent ranking ability; the operating threshold should be set by business requirements, not by optimising F1.

5. **RL-based feature exploration confirms supervised rankings**: The RL agent independently identified the same top features as the supervised importance analysis, providing cross-validation of the feature importance findings.

---

## 10. Limitations and Future Work

### 10.1 Limitations

- **Low recall at default threshold (7%)**: The deployed threshold of 0.5 is conservative. Operational deployment should use a lower threshold (0.1–0.3) based on the cost ratio of missed fraud vs. false positives.
- **Static terminal statistics**: Terminal fraud rates are computed at training time. In production, these should be updated continuously as new transactions arrive.
- **Simulated data**: The dataset is synthetic. Real-world performance may differ due to adversarial fraud patterns not present in simulation.
- **No concept drift detection**: The model has no mechanism to detect when the fraud pattern distribution shifts over time.

### 10.2 Future Work

1. **Online learning**: Update terminal statistics and model weights in real time using streaming data (Apache Kafka + MLflow).
2. **Threshold optimisation**: Use the precision-recall curve to select a threshold that minimises expected financial loss (requires cost-benefit analysis with fraud loss vs. false alarm cost).
3. **Graph neural networks**: Model the customer-terminal transaction network as a graph to capture relationship-based fraud patterns.
4. **Ensemble stacking**: Combine XGBoost, LightGBM, and a neural network via a meta-learner.
5. **Explainability**: Integrate SHAP values into the dashboard to show per-transaction feature contributions.

---

## 11. Conclusion

This work demonstrates a complete, production-deployed fraud detection system built from raw relational data. The XGBoost model achieves a ROC-AUC of 0.9518 through principled feature engineering, particularly the discovery that terminal history accounts for 58% of predictive power. The system is deployed live on Azure App Service with a real-time monitoring dashboard, completing the full ML lifecycle from data exploration to production serving.

The primary contribution is the systematic feature engineering methodology — especially the personalised amount deviation features and fitted terminal risk statistics — which can be applied to any transaction fraud detection problem regardless of the underlying model architecture.

---

## References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
2. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
3. Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
4. Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR, 16*, 321–357.
5. Nilson Report (2023). Payment card fraud losses worldwide.
6. Dal Pozzolo, A. et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *SSCI 2015*.

---

*This report was produced as part of the Wits AI/ML Learning Course. All experiments are reproducible from the linked repository.*
