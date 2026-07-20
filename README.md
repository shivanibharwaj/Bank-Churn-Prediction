# Customer Bank Churn Prediction using Machine Learning

## Project Overview

This project predicts whether a bank customer is likely to leave the bank (churn) or stay using Machine Learning techniques. The goal is to help banks identify customers who are at risk of leaving so they can take actions to improve customer retention.

---

## Problem Statement

Customer churn is one of the major challenges faced by banks. By predicting which customers are likely to leave, banks can take preventive measures such as personalized offers and better customer support.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## Dataset

The project uses the **Churn Modelling Dataset**.

### Features:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

### Target Variable
- Exited (0 = Customer Stayed, 1 = Customer Churned)

---

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Data Visualization
5. Feature Encoding
6. Train-Test Split
7. Feature Scaling
8. Model Training
9. Model Evaluation
10. Model Saving using Joblib

---

## Machine Learning Models Used

### Logistic Regression
- Accuracy: **81.1%**

### Random Forest Classifier
- Accuracy: **86.65%**

Random Forest performed better and was selected as the final model.

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

## Project Structure

```
bank_churn_prediction/
│
├── app/
├── data/
├── models/
├── notebooks/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Results

- Successfully predicted customer churn.
- Random Forest achieved better performance than Logistic Regression.
- Final Accuracy: **86.65%**

---

## Future Improvements

- Hyperparameter Tuning
- XGBoost Model
- SMOTE for handling class imbalance
- Web Application using Streamlit or Flask
- Model Deployment

---

## Author

**Shivani Kumari**

B.Tech - Computer Science & Engineering (Data Science)

Haldia Institute of Technology
