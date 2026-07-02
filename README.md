# 🏦 Customer Churn Prediction in Banking

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)

### 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

**A Complete End-to-End Machine Learning Project**

*Predicting customer churn in banking using advanced ML algorithms*

---

[🌐 Live App](https://bank-churn-prediction-9ma6yauphxnrsvkj5kwwfg.streamlit.app/) •
[📊 Dataset](#-dataset) •
[🤖 Models](#-models) •
[📈 Results](#-results) •


</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Live Demo](#-live-demo)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Models](#-models)
- [Results](#-results)


---

## 🎯 Overview

Customer churn prediction is one of the most critical challenges in the banking industry. This project builds a **complete end-to-end machine learning pipeline** that:

- 📊 Analyzes **10,000** bank customer records
- 🔍 Performs comprehensive **Exploratory Data Analysis**
- ⚙️ Implements advanced **feature engineering**
- 🤖 Trains and compares **10 ML algorithms**
- 🌐 Deploys as an interactive **Streamlit web application**
- 🔮 Provides **real-time churn predictions** with actionable recommendations

> 💡 **Key Insight:** Acquiring a new customer costs **5-25x more** than retaining an existing one. A 5% increase in retention can boost profits by **25-95%!**

---

## ❓ Problem Statement

Banks lose billions annually due to customer attrition. Traditional methods are:

- ❌ **Reactive** — too late to act
- ❌ **Inefficient** — manual and slow process
- ❌ **Inaccurate** — based on intuition not data

✅ **Our Solution:** A machine learning model that predicts which customers will churn **before** they leave, enabling proactive retention strategies.

---

## 🌐 Live Demo

🔗 **[Click Here to Open the Live App](https://bank-churn-prediction-9ma6yauphxnrsvkj5kwwfg.streamlit.app/)**

The app includes **8 interactive pages:**

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview and problem statement |
| 📊 Dataset Overview | Data preview, statistics, missing values |
| 🔍 EDA | Interactive charts and analysis |
| ⚙️ Preprocessing | Data cleaning and feature engineering |
| 🤖 Model Training | Train 10 ML models with full metrics |
| 📈 Model Comparison | Compare all models side by side |
| 🔮 Predict Churn | Real-time customer churn prediction |
| 📋 Summary | Conclusions and recommendations |

---

## 📁 Project Structure

🏦 BankChurnProject/
│
├── 🏠 Home.py ← Main Streamlit application
├── 📋 requirements.txt ← Python dependencies
├── 🚫 .gitignore ← Git ignore rules
├── 🔧 run.bat ← Windows one-click launcher
│
├── 📁 .streamlit/
│ └── ⚙️ config.toml ← Streamlit configuration
│
├── 📁 data/
│ ├── 🐍 init.py
│ └── 🐍 generate_data.py ← Dataset generation module
│
└── 📁 src/
├── 🐍 init.py
├── 🐍 preprocessing.py ← Data preprocessing pipeline
├── 🐍 model_training.py ← ML model training and evaluation
└── 🐍 predictor.py ← Real-time prediction module

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| 📦 **Total Records** | 10,000 customers |
| 📋 **Original Features** | 14 |
| 🔧 **Engineered Features** | 6 |
| 🎯 **Target Variable** | Exited (0=Stayed, 1=Churned) |
| ⚖️ **Churn Rate** | ~20% |
| 📉 **Class Imbalance** | 4:1 (Stayed:Churned) |
| 🌍 **Geographies** | France, Spain, Germany |

### 📋 Original Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| CreditScore | Numerical | Customer credit score | 350 - 850 |
| Geography | Categorical | Country of residence | France / Spain / Germany |
| Gender | Categorical | Customer gender | Male / Female |
| Age | Numerical | Customer age | 18 - 92 |
| Tenure | Numerical | Years with the bank | 0 - 10 |
| Balance | Numerical | Account balance | 0 - 260,000 |
| NumOfProducts | Numerical | Bank products used | 1 - 4 |
| HasCrCard | Binary | Has credit card | 0 / 1 |
| IsActiveMember | Binary | Active member status | 0 / 1 |
| EstimatedSalary | Numerical | Annual salary | 10 - 200,000 |
| **Exited** | **Binary** | **TARGET: Churned** | **0 / 1** |

### 🔧 Engineered Features

| New Feature | Formula | Rationale |
|-------------|---------|-----------|
| BalanceSalaryRatio | Balance / Salary | Financial engagement relative to income |
| TenureAgeRatio | Tenure / Age | Customer loyalty vs age |
| CreditScoreAge | CreditScore / Age | Creditworthiness vs age |
| HasBalance | Balance > 0 | Whether customer maintains balance |
| ProductsPerTenure | Products / (Tenure+1) | Product adoption rate |
| AgeGroup | Binned Age 0-4 | Age based segmentation |

---

## 🤖 Models

We trained and compared **10 Machine Learning algorithms:**

| # | Algorithm | Type | Key Strength |
|---|-----------|------|-------------|
| 1 | Logistic Regression | Linear | Interpretable baseline |
| 2 | Decision Tree | Tree-based | Non-linear, visual |
| 3 | Random Forest | Ensemble Bagging | Reduces overfitting |
| 4 | Gradient Boosting | Ensemble Boosting | Sequential correction |
| 5 | ⭐ **XGBoost** | **Ensemble Boosting** | **Best performer** |
| 6 | LightGBM | Ensemble Boosting | Fast training |
| 7 | K-Nearest Neighbors | Instance-based | Distance classification |
| 8 | Support Vector Machine | Kernel-based | High dimensional data |
| 9 | Naive Bayes | Probabilistic | Fast and simple |
| 10 | AdaBoost | Ensemble Boosting | Adaptive learning |

### 🔄 Complete ML Pipeline
Raw Data
↓
Data Cleaning
↓
Feature Engineering (6 new features)
↓
Encoding (Label + One-Hot)
↓
Train-Test Split (80/20 Stratified)
↓
Feature Scaling (StandardScaler)
↓
SMOTE (Handle Class Imbalance)
↓
Model Training (10 Algorithms)
↓
Evaluation (5 Metrics)
↓
Deployment (Streamlit App)

---

## 📈 Results

### 🏆 Model Performance Comparison

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 🥇 1 | **XGBoost** | **0.8530** | **0.6215** | **0.7312** | **0.6720** | **0.8745** |
| 🥈 2 | LightGBM | 0.8495 | 0.6148 | 0.7389 | 0.6713 | 0.8698 |
| 🥉 3 | Gradient Boosting | 0.8510 | 0.6192 | 0.7205 | 0.6660 | 0.8672 |
| 4 | Random Forest | 0.8545 | 0.6340 | 0.6856 | 0.6588 | 0.8590 |
| 5 | AdaBoost | 0.8420 | 0.5980 | 0.7105 | 0.6495 | 0.8512 |
| 6 | Logistic Regression | 0.7890 | 0.5245 | 0.7520 | 0.6178 | 0.8345 |
| 7 | SVM | 0.8115 | 0.5610 | 0.7150 | 0.6290 | 0.8290 |
| 8 | KNN | 0.7985 | 0.5412 | 0.6890 | 0.6062 | 0.7980 |
| 9 | Naive Bayes | 0.7320 | 0.4580 | 0.7810 | 0.5775 | 0.8120 |
| 10 | Decision Tree | 0.7810 | 0.5120 | 0.6542 | 0.5745 | 0.7645 |

### 🎯 Top 10 Important Features (XGBoost)

| Rank | Feature | Importance | Category |
|------|---------|------------|---------|
| 🥇 1 | Age | 0.1845 | Original |
| 🥈 2 | NumOfProducts | 0.1523 | Original |
| 🥉 3 | IsActiveMember | 0.1198 | Original |
| 4 | Balance | 0.0987 | Original |
| 5 | Geography_Germany | 0.0876 | Encoded |
| 6 | CreditScoreAge | 0.0654 | Engineered |
| 7 | BalanceSalaryRatio | 0.0598 | Engineered |
| 8 | AgeGroup | 0.0512 | Engineered |
| 9 | Gender | 0.0489 | Original |
| 10 | CreditScore | 0.0423 | Original |

### 🔑 Key EDA Findings
📊 Dataset is IMBALANCED → 80% stayed vs 20% churned
👴 Age is number 1 predictor → Customers aged 40-60 churn most
🌍 Germany → Highest churn rate 32%
👫 Gender gap → Females 25% vs Males 16%
💳 Products → 3-4 products = very high churn
🔄 Activity → Inactive members churn 2x more
