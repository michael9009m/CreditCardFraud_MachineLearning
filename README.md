
MICHAEL MARTINEZ: FRADULENT TRANSACTION CHECKER

LINK TO THE DATASETt:https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

THE PROBLEM:
    You are contacted by a European bank to create a Machine Learning Model that can detect fradulent transactions with high accuracy while not inconvincing customers that are doing legitamate purchases with fraud alerts or frozen cards. In this data set there are 284,807 total transactions and only 492 of them are fraudulent. That means that only 0.172% of all transactions are fraud.
        Total transactions:284,807
        Legitamate transactions: 99.828% 
        Fraudlent transactions: 0.172%

WHAT AM I LOOKING FOR:
    With a problem of this scale and difficulty, it is important to know what exactly you are looking for. The three main factors I considered were Precision, Recall, and F1 score. 
        -Precision: How many transactions that the model said was fradulent, was ACTUALLY fraudulent
        -Recall: Out of all of the fraudulent transactions how many did you mark as fraudulent
        -F1 score: How well the model is overall with precision and recall combined

THE PROCESS:


MY SOLUTION:
    I tested five different machine learning models to see which would give the most accurate results. I decided on testing Logistic Regression, A Shallow Neural Network, Random Forrest, Gradient Boost, Linear SVM. 

IMBALANCED DATA
 
                        P         R        F1
Logistic R on val:    0.83      0.56      0.67 
Shallow_nn on val:    0.65      0.78      0.71    
Random For on val:    0.81      0.47      0.60
Grad Boost on val:    0.67      0.67      0.67
Linear SVM on val:    0.77      0.75      0.76 (Had to manually balance to keep realistic)


BALANCED DATA
Log Reg
      Not Fraud       0.94      0.94      0.94        
          Fraud       0.95      0.95      0.95

Shallow nn: 2 relu
.     Not Fraud       0.81      1.00      0.89       
          Fraud       1.00      0.78      0.88

Shallow nn: 1 relu
.     Not Fraud       0.81      1.00      0.89        
          Fraud       1.00      0.78      0.88

Random Forrest
     Not Fraud       0.65      1.00      0.79        
         Fraud       1.00      0.50      0.67

Gradient Boost
     Not Fraud       0.72      1.00      0.84        
         Fraud       1.00      0.65      0.79

SVM
     Not Fraud       0.67      1.00      0.80        
         Fraud       1.00      0.54      0.70

#TESTED NEURAL NET
     Not Fraud       0.85      1.00      0.92        
         Fraud       1.00      0.81      0.90

