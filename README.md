# Financial Fraud Detection using Machine Learning

## Project Overview
This project focuses on detecting fraudulent financial transactions using machine learning techniques. The dataset includes transaction and identity information, which are merged and processed to train classification models capable of identifying fraudulent activity.

Fraud detection is a challenging problem because the number of fraudulent transactions is significantly smaller than legitimate ones, making the dataset highly imbalanced.

---

## Objectives
- Detect fraudulent financial transactions accurately  
- Handle highly imbalanced datasets using resampling techniques  
- Compare the performance of different machine learning models  

---

## Dataset
The dataset contains:

- **Transaction Data** – Information about each financial transaction  
- **Identity Data** – Additional user and device identity attributes  

These datasets are merged using the `TransactionID` field.

### Target Variable

`isFraud`

- **0 → Legitimate Transaction**
- **1 → Fraudulent Transaction**

---

## Project Workflow

1. **Data Merging** – Combined transaction and identity datasets  
2. **Data Preprocessing** – Handled missing values and inconsistent data  
3. **Feature Encoding** – Applied label encoding for categorical variables  
4. **Handling Class Imbalance** – Used **SMOTE** to balance fraud and non-fraud samples  
5. **Model Training** – Trained machine learning models on processed data  
6. **Model Evaluation** – Evaluated models using ROC-AUC and classification metrics  
7. **Prediction Generation** – Generated fraud probability predictions for test data  

---

## Machine Learning Models Used

- **Random Forest Classifier**
- **XGBoost Classifier**

---

## Evaluation Metrics

The models were evaluated using:

- ROC-AUC Score  
- Precision  
- Recall  
- F1 Score  

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  

## Model Performance

| Model | ROC-AUC Score |
|------|---------------|
| Random Forest | 0.94 |
| XGBoost | 0.97 |
---

## Output

The trained model generates fraud probability predictions saved in:

`submission.csv`

This file contains predicted fraud probabilities for each transaction in the test dataset.

---

---

## Visualizations

### Fraud vs Non-Fraud Distribution

<p align="center">
  <img src="fraud vs non-fraud distribution.png" width="600">
</p>

This chart shows the imbalance between legitimate and fraudulent transactions in the dataset.

---

### Transaction Amount vs Fraud

<p align="center">
  <img src="box-plot_Transaction Amount vs Fraud.png" width="600">
</p>

The boxplot compares transaction amounts for fraudulent and non-fraudulent transactions.

---

### ROC Curve Comparison

<p align="center">
  <img src="ROC curve comparision.png" width="600">
</p>

The ROC curve compares the performance of Random Forest and XGBoost models.

---

## Author
**Simham Vijay**
