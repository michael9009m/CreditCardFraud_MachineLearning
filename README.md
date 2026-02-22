Here is a clean, professional GitHub-ready README version. You can paste this directly into `README.md`.

---

# Fraudulent Transaction Checker

### Michael Martinez

## Dataset

Kaggle Credit Card Fraud Dataset
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Dataset Overview**

* Total transactions: 284,807
* Legitimate transactions: 99.828%
* Fraudulent transactions: 0.172% (492 transactions)

This is an extreme class imbalance problem. Only 0.172% of transactions are fraud, which makes traditional accuracy misleading and unreliable.

---

## The Problem

A European bank needs a machine learning model that:

* Detects fraudulent transactions with high accuracy
* Minimizes false fraud alerts on legitimate customers
* Maintains a strong balance between precision and recall

The challenge is catching rare fraud cases without disrupting legitimate customer activity.

---

## Evaluation Metrics

To properly evaluate performance, I focused on:

**Precision**
How many transactions predicted as fraud were actually fraud.

**Recall**
Out of all real fraud transactions, how many were correctly identified.

**F1 Score**
A balance between precision and recall.

In fraud detection:

* High precision reduces unnecessary customer friction.
* High recall reduces financial loss.
* F1 score balances both.

---

## Models Tested

* Logistic Regression
* Shallow Neural Network
* Random Forest
* Gradient Boosting
* Linear SVM

Each model was tested on both the original imbalanced dataset and a manually balanced version.

---

# Results

## Imbalanced Data (Real-World Distribution)

| Model               | Precision | Recall | F1 Score |
| ------------------- | --------- | ------ | -------- |
| Logistic Regression | 0.83      | 0.56   | 0.67     |
| Shallow Neural Net  | 0.65      | 0.78   | 0.71     |
| Random Forest       | 0.81      | 0.47   | 0.60     |
| Gradient Boost      | 0.67      | 0.67   | 0.67     |
| Linear SVM          | 0.77      | 0.75   | 0.76     |

Linear SVM achieved the strongest overall balance on the imbalanced validation set.

---

## Balanced Data Results

### Logistic Regression

| Class     | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| Not Fraud | 0.94      | 0.94   | 0.94     |
| Fraud     | 0.95      | 0.95   | 0.95     |

---

### Shallow Neural Network (2 ReLU Layers)

| Class     | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| Not Fraud | 0.81      | 1.00   | 0.89     |
| Fraud     | 1.00      | 0.78   | 0.88     |

---

### Shallow Neural Network (1 ReLU Layer)

| Class     | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| Not Fraud | 0.81      | 1.00   | 0.89     |
| Fraud     | 1.00      | 0.78   | 0.88     |

---

### Random Forest

| Class     | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| Not Fraud | 0.65      | 1.00   | 0.79     |
| Fraud     | 1.00      | 0.50   | 0.67     |

---

### Gradient Boosting

| Class     | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| Not Fraud | 0.72      | 1.00   | 0.84     |
| Fraud     | 1.00      | 0.65   | 0.79     |

---

### Support Vector Machine (SVM)

| Class     | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| Not Fraud | 0.67      | 1.00   | 0.80     |
| Fraud     | 1.00      | 0.54   | 0.70     |

---

## Final Tested Neural Network (Test Set)

| Class     | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| Not Fraud | 0.85      | 1.00   | 0.92     |
| Fraud     | 1.00      | 0.81   | 0.90     |

The final neural network achieved:

* Perfect fraud precision on the test set
* Strong fraud recall (81%)
* High overall F1 performance
* A practical balance between minimizing false alerts and detecting fraud

---

## Conclusion

This project demonstrates:

* The difficulty of extreme class imbalance problems
* Why accuracy is not sufficient in fraud detection
* The importance of balancing precision and recall
* That properly tuned neural networks can outperform traditional models in this domain

The final neural network provided the strongest real-world balance and is the best candidate for deployment.

---

If you want, I can also tighten this further to make it look even more recruiter-friendly and less academic.
