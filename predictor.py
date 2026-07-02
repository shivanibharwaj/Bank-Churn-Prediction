import numpy as np


def predict_customer(model, scaler, customer_data):
    credit_score = customer_data["CreditScore"]
    gender_enc = 1 if customer_data["Gender"] == "Male" else 0
    age = customer_data["Age"]
    tenure = customer_data["Tenure"]
    balance = customer_data["Balance"]
    num_products = customer_data["NumOfProducts"]
    has_cr_card_enc = 1 if customer_data["HasCrCard"] == "Yes" else 0
    is_active_enc = 1 if customer_data["IsActiveMember"] == "Yes" else 0
    salary = customer_data["EstimatedSalary"]
    geo_germany = 1 if customer_data["Geography"] == "Germany" else 0
    geo_spain = 1 if customer_data["Geography"] == "Spain" else 0
    bal_sal_ratio = balance / (salary + 1)
    ten_age_ratio = tenure / (age + 1)
    cs_age = credit_score / (age + 1)
    has_bal = 1 if balance > 0 else 0
    prod_tenure = num_products / (tenure + 1)
    if age <= 30: age_group = 0
    elif age <= 40: age_group = 1
    elif age <= 50: age_group = 2
    elif age <= 60: age_group = 3
    else: age_group = 4
    features = np.array([[credit_score, gender_enc, age, tenure, balance, num_products, has_cr_card_enc, is_active_enc, salary, bal_sal_ratio, ten_age_ratio, cs_age, has_bal, prod_tenure, age_group, geo_germany, geo_spain]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    return prediction, probability