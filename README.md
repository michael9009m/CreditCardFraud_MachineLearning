# Fraudulent Transaction Checker
**Michael Martinez**

A real-time credit card fraud detection system built with XGBoost, deployed serverlessly on AWS Lambda, and served through a live interactive web demo.

🔗 **[Live Demo](https://michael9009m.github.io/CreditCardFraud_MachineLearning/)**

---

## Overview

This project tackles one of the hardest problems in applied machine learning — detecting fraud in an extremely imbalanced dataset where only 0.172% of transactions are fraudulent. A traditional accuracy metric would be meaningless here (a model that predicts "legitimate" every time achieves 99.8% accuracy). Instead, the focus is on precision, recall, and F1 score to build something that works in the real world.

The final model is deployed as a serverless AWS Lambda function behind an API Gateway endpoint, callable from anywhere on the internet in real time.

---

## Dataset

**Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Stat | Value |
|---|---|
| Total transactions | 284,807 |
| Legitimate | 99.828% |
| Fraudulent | 0.172% (492 cases) |

Features V1–V28 are the result of PCA transformation applied by the original bank for privacy. Time and Amount are the only raw features.

---

## The Problem

A European bank needs a model that:
- Detects fraudulent transactions with high accuracy
- Minimizes false fraud alerts on legitimate customers
- Maintains a strong balance between precision and recall
- Can run inference in real time on new, unseen transactions

The challenge is catching rare fraud cases without disrupting legitimate customer activity.

---

## Why Accuracy Is The Wrong Metric

On a dataset where 99.8% of transactions are legitimate, a model that always predicts "legitimate" achieves 99.8% accuracy — and catches zero fraud. This project focuses on:

- **Precision** — of all transactions flagged as fraud, how many actually are?
- **Recall** — of all real fraud cases, how many did the model catch?
- **F1 Score** — the harmonic mean of precision and recall, balancing both

---

## Model Selection

Five models were evaluated on both the original imbalanced dataset and a manually balanced version (equal fraud and legitimate samples):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.83 | 0.56 | 0.67 |
| Shallow Neural Network | 0.65 | 0.78 | 0.71 |
| Random Forest | 0.81 | 0.47 | 0.60 |
| Gradient Boosting | 0.67 | 0.67 | 0.67 |
| Linear SVM | 0.77 | 0.75 | 0.76 |

After evaluating all five, **XGBoost** was selected for the final model. XGBoost is the industry standard for tabular fraud detection — used by companies like PayPal and Stripe — and outperformed all other models on the balanced dataset while remaining lightweight enough to deploy serverlessly.

---

## Final Model Results (XGBoost)

### Validation Set
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not Fraud | 0.96 | 0.90 | 0.93 |
| Fraud | 0.91 | 0.96 | 0.93 |

### Test Set
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not Fraud | 0.91 | 0.93 | 0.92 |
| Fraud | 0.93 | 0.90 | 0.91 |

**Overall test accuracy: 92%**

---

## Preprocessing

- `Amount` scaled using **RobustScaler** (robust to outliers from extreme transaction values)
- `Time` normalized using **min-max scaling** across the full dataset
- `V1–V28` passed through as-is (already PCA-transformed)
- Dataset manually balanced by undersampling legitimate transactions to match fraud count before training

All preprocessing parameters are saved to `scaler_params.json` and applied identically at inference time inside the Lambda function.

---

## Architecture

```
Browser (demo webpage)
        │
        │  POST /predict  (JSON payload)
        ▼
AWS API Gateway
        │
        ▼
AWS Lambda (Python 3.11)
  ├── Loads XGBoost model (fraud_model_xgb.json)
  ├── Applies preprocessing (scaler_params.json)
  ├── Runs inference
  └── Returns prediction + probability
```

---

## Live Demo Features

The demo webpage provides two modes:

**Preset Transactions**
Two real rows pulled directly from the Kaggle dataset — one known legitimate transaction and one known fraud case. Ground truth is known, so the model's prediction can be verified instantly.

**Synthetic Generator**
Generates statistically realistic synthetic transactions by sampling V1–V28 from the actual mean and standard deviation of either the fraud class or the legitimate class in the training data (using Box-Muller transform). This simulates what production inference actually looks like — a brand new transaction the model has never seen, with no ground truth. The user selects a class distribution and a transaction amount, and the model returns a real-time prediction from the live Lambda endpoint.

> Note: Synthetic transactions sample each V feature independently, which does not capture inter-feature correlations present in real fraud patterns. This occasionally produces misclassifications, which is expected and intentional — it demonstrates the model's behavior on statistically plausible but imperfect inputs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost |
| Training | Python, scikit-learn, pandas |
| Inference | AWS Lambda (Python 3.11) |
| API | AWS API Gateway (HTTP API) |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Hosting | GitHub Pages |
| Model format | XGBoost native JSON |
| Dependencies | xgboost, numpy, scipy |

---

## Project Structure

```
CreditCardFraud_MachineLearning/
├── detectingCreditCardFraud.ipynb   # Full training notebook
├── fraud_model_xgb.json             # Trained XGBoost model
├── scaler_params.json               # Preprocessing parameters
├── class_stats.json                 # Per-class feature distributions
├── lambda_function.py               # AWS Lambda handler
├── index.html                       # Live demo webpage
└── README.md
```

---

## Key Takeaways

- Accuracy is not a valid metric for highly imbalanced classification problems
- XGBoost consistently outperforms neural networks on tabular fraud detection data
- Serverless deployment via AWS Lambda enables real-time inference at near-zero idle cost
- Synthetic data generation using class-conditional distributions can simulate production inference, but inter-feature correlations matter for realistic results
