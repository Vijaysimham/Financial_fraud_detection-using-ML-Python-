# 💳 Financial Fraud Detection using Machine Learning

## 📌 Project Overview
Financial fraud is a major challenge for financial institutions and online payment systems. Detecting fraudulent transactions quickly and accurately is essential to minimize financial losses and protect customers.

This project implements a **machine learning pipeline for financial fraud detection** using transaction and identity data. The workflow includes data preprocessing, handling missing values, encoding categorical variables, addressing class imbalance using **SMOTE**, and training machine learning models including **Random Forest** and **XGBoost**.

The trained model predicts fraudulent transactions and generates a **submission-ready output file**.

---

# 📂 Dataset

This project uses two primary datasets:

| Dataset | Description |
|-------|-------------|
| `train_transaction.csv` | Contains transaction-level financial information |
| `train_identity.csv` | Contains identity-related details of users |

Both datasets are merged using:

```
TransactionID
```

This creates a combined dataset containing both **transaction features and identity features**.

---

# ⚙️ Machine Learning Pipeline

The project follows a structured **end-to-end machine learning workflow**.

## 1️⃣ Data Loading

Transaction and identity datasets are loaded and merged.

```python
train_trans = pd.read_csv("train_transaction.csv")
train_id = pd.read_csv("train_identity.csv")

data = train_trans.merge(train_id, on="TransactionID", how="left")
```

---

# 🧹 Data Preprocessing

## Handling Missing Values

Missing values are handled using different strategies depending on data type.

- **Categorical columns:** filled with `"Unknown"`
- **Numerical columns:** filled with **median values**

```python
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna("Unknown", inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)
```

---

## Encoding Categorical Variables

Machine learning models require numerical input, so categorical variables are encoded using **Label Encoding**.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])
```

---

# 📊 Exploratory Data Analysis

Exploratory analysis helps understand the dataset.

### Fraud Distribution
Shows the number of fraudulent vs legitimate transactions.

### Transaction Amount Analysis
Visualizes how transaction amounts differ between fraud and non-fraud transactions.

These visualizations help understand **class imbalance and fraud patterns**.

---

# ⚖️ Handling Class Imbalance

Fraud datasets are usually **highly imbalanced**, meaning fraudulent transactions are much fewer than legitimate ones.

To address this, **SMOTE (Synthetic Minority Oversampling Technique)** is applied.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

This helps the model learn patterns from **minority fraud cases**.

---

# 🤖 Model Training

Two machine learning models were implemented and compared.

---

## 🌲 Random Forest

Random Forest is an ensemble model based on multiple decision trees.

```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

### Advantages
- Handles high dimensional datasets
- Reduces overfitting
- Provides feature importance scores

---

## ⚡ XGBoost

XGBoost is a powerful **gradient boosting algorithm** widely used in machine learning competitions and real-world applications.

```python
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10,
    random_state=42
)
```

### Advantages
- High predictive performance
- Efficient handling of large datasets
- Built-in regularization

---

# 📈 Model Evaluation

Models are evaluated using:

- **Classification Report**
- **ROC-AUC Score**
- **ROC Curve**

```python
roc_auc_score(y_test, rf_probs)
roc_auc_score(y_test, xgb_probs)
```

The ROC curve helps visualize model performance by comparing **True Positive Rate vs False Positive Rate**.

---

# 🔎 Feature Importance

Feature importance helps identify the most influential variables for fraud detection.

```python
feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
```

This provides insights into **which transaction attributes contribute most to detecting fraud**.

---

# 🧪 Prediction on Test Data

The trained XGBoost model is used to generate predictions on the test dataset.

Steps include:

1. Loading test transaction data
2. Merging with identity dataset
3. Aligning features with training dataset
4. Generating fraud probability predictions

```python
test_preds = xgb.predict_proba(test_data)[:, 1]
```

---

# 📤 Submission File Generation

The predictions are saved as a submission file.

```python
submission = pd.DataFrame({
    "TransactionID": test_trans["TransactionID"],
    "isFraud": test_preds
})

submission.to_csv("submission.csv", index=False)
```




---

# 🛠 Technologies Used

| Category | Tools |
|--------|------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Imbalance Handling | SMOTE |
| Advanced Model | XGBoost |

---

# 🎯 Skills Demonstrated

- Data preprocessing
- Handling missing values
- Feature encoding
- Class imbalance handling
- Machine learning model development
- Model evaluation
- Feature importance analysis
- Fraud detection modeling
- End-to-end ML pipeline implementation

---

# 🚀 Future Improvements

Possible improvements for the project include:

- Advanced feature engineering
- Hyperparameter tuning using GridSearchCV
- Deep learning models
- Real-time fraud detection systems
- Deployment using **Flask or FastAPI**
- Model monitoring and performance tracking

---

# 👨‍💻 Author

**Simham Vijay**

BCA Graduate
Aspiring Data Analyst
