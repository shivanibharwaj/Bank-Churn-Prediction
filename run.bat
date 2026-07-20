@echo off
echo ============================================
echo   Bank Customer Churn Prediction Project
echo ============================================
echo.
echo Installing dependencies...
pip install streamlit pandas numpy scikit-learn xgboost lightgbm imbalanced-learn plotly seaborn matplotlib joblib openpyxl
echo.
echo Starting application...
cd /d "C:\Users\Aryan\Desktop\project\BankChurnProject"
streamlit run Home.py
pause
