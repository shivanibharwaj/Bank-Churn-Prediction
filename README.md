# Customer Bank Churn Prediction

## Project Overview

Customer churn is when a customer stops using the services of a company or bank. 
The goal of this project is to predict whether a bank customer is likely to leave the bank or continue using its services.

This project uses Machine Learning classification algorithms to analyze customer data and predict churn.

---

## Problem Statement

Banks lose revenue when customers leave their services. 
By predicting potential churn customers in advance, banks can take preventive actions such as providing offers and improving customer experience.

---

## Dataset Information

Dataset: Bank Customer Churn Dataset

The dataset contains customer information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card Status
- Active Membership Status
- Estimated Salary

Target Variable:

- Exited
    - 0 → Customer stayed
    - 1 → Customer left

---

## Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Tools
- Jupyter Notebook
- GitHub
- VS Code

---

## Machine Learning Workflow

The project follows these steps:

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Encoding
6. Feature Scaling
7. Model Training
8. Model Evaluation
9. Model Saving

---

## Machine Learning Models Used

The following classification algorithms were considered:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Final Model:

Random Forest Classifier

Reason:
Random Forest provides better performance by combining multiple decision trees and reducing overfitting.

---

## Model Evaluation Metrics

The model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Project Structure
bank-churn-prediction/

│
├── data/
│ └── Churn_Modelling.csv
│
├── models/
│ ├── customer_churn_model.pkl
│ └── scaler.pkl
│
├── notebooks/
│ └── 01_EDA.ipynb
│
├── requirements.txt
│
└── README.md

---

## How to Run the Project

Clone the repository:

Install required libraries:

---

## Future Improvements

- Deploy the model using Streamlit
- Add real-time customer prediction
- Improve model performance using advanced algorithms

---
## How to Run the Streamlit Application

### 1. Clone the Repository

```bash
git clone https://github.com/shivanibharwaj/customer-bank-churn-prediction.git
```

### 2. Navigate to Project Folder

```bash
cd customer-bank-churn-prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser where users can enter customer details and get churn predictions.

---

## Author

**Shivani Kumari**

B.Tech - Computer Science & Engineering (Data Science)

Haldia Institute of Technology
