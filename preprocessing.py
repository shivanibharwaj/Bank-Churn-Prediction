import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def preprocess_data(df):
    df_processed = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    df_processed["BalanceSalaryRatio"] = df_processed["Balance"] / (df_processed["EstimatedSalary"] + 1)
    df_processed["TenureAgeRatio"] = df_processed["Tenure"] / (df_processed["Age"] + 1)
    df_processed["CreditScoreAge"] = df_processed["CreditScore"] / (df_processed["Age"] + 1)
    df_processed["HasBalance"] = (df_processed["Balance"] > 0).astype(int)
    df_processed["ProductsPerTenure"] = df_processed["NumOfProducts"] / (df_processed["Tenure"] + 1)
    df_processed["AgeGroup"] = pd.cut(df_processed["Age"], bins=[0,30,40,50,60,100], labels=[0,1,2,3,4]).astype(int)
    le = LabelEncoder()
    df_processed["Gender"] = le.fit_transform(df_processed["Gender"])
    df_processed = pd.get_dummies(df_processed, columns=["Geography"], drop_first=True, dtype=int)
    return df_processed


def prepare_training_data(df_processed):
    X = df_processed.drop("Exited", axis=1)
    y = df_processed["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    return X_train_res, X_test_scaled, y_train_res, y_test, scaler, X.columns.tolist()