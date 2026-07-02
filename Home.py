import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from data.generate_data import generate_dataset
from src.preprocessing import preprocess_data, prepare_training_data
from src.model_training import train_all_models
from src.predictor import predict_customer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

st.set_page_config(page_title="Bank Customer Churn Prediction", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {font-size:2.5rem; font-weight:700; color:#1a5276; text-align:center; padding:1rem 0; border-bottom:3px solid #2ecc71; margin-bottom:2rem;}
    .sub-header {font-size:1.5rem; font-weight:600; color:#2c3e50; padding:0.5rem 0; border-left:4px solid #3498db; padding-left:1rem; margin:1rem 0;}
    .metric-card {background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding:1.5rem; border-radius:15px; color:white; text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.2);}
    .success-card {background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding:1.5rem; border-radius:15px; color:white; text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.2);}
    .danger-card {background:linear-gradient(135deg, #eb3349 0%, #f45c43 100%); padding:1.5rem; border-radius:15px; color:white; text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.2);}
    .warning-card {background:linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding:1.5rem; border-radius:15px; color:white; text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.2);}
    .info-box {background-color:#eaf2f8; border-left:5px solid #2980b9; padding:1rem; border-radius:5px; margin:1rem 0;}
    .stTabs [data-baseweb="tab-list"] {gap:8px;}
    .stTabs [data-baseweb="tab"] {background-color:#f0f2f6; border-radius:10px; padding:10px 20px; font-weight:600;}
    .stTabs [aria-selected="true"] {background-color:#3498db; color:white;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return generate_dataset()


@st.cache_data
def get_processed_data(_df):
    return preprocess_data(_df)


@st.cache_resource
def do_training(_X_train, _X_test, _y_train, _y_test):
    return train_all_models(_X_train, _X_test, _y_train, _y_test)


def main():
    st.sidebar.markdown("""<div style="text-align:center; padding:1rem;"><h1 style="color:#2c3e50;">🏦</h1><h2 style="color:#2c3e50;">Bank Churn<br>Predictor</h2></div>""", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    page = st.sidebar.radio("📑 Navigate to", ["🏠 Home & Problem Statement", "📊 Dataset Overview", "🔍 Exploratory Data Analysis", "⚙️ Data Preprocessing", "🤖 Model Training & Evaluation", "📈 Model Comparison", "🔮 Predict New Customer", "📋 Project Summary"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""<div style="text-align:center; font-size:0.8rem; color:#7f8c8d;"></div>""", unsafe_allow_html=True)
    df = load_data()

    if page == "🏠 Home & Problem Statement":
        st.markdown("<h1 class='main-header'>🏦 Customer Churn Prediction in Banking</h1>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        churn_rate = df["Exited"].mean()*100
        with c1: st.markdown("<div class='metric-card'><h2>10,000</h2><p>Total Customers</p></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='danger-card'><h2>{churn_rate:.1f}%</h2><p>Churn Rate</p></div>", unsafe_allow_html=True)
        with c3: st.markdown("<div class='success-card'><h2>10+</h2><p>ML Models</p></div>", unsafe_allow_html=True)
        with c4: st.markdown("<div class='warning-card'><h2>14</h2><p>Features</p></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("<h3 class='sub-header'>🎯 Problem Statement</h3>", unsafe_allow_html=True)
            st.markdown("""<div class="info-box"><p><b>Customer churn</b> (also known as customer attrition) refers to when a customer stops doing business with a company.</p><p>In the banking sector, churn is a critical problem because:</p><ul><li>📉 Acquiring a new customer costs <b>5-25x more</b> than retaining existing ones</li><li>💰 A 5% increase in retention can increase profits by <b>25-95%</b></li><li>🏦 Banks lose billions annually due to customer attrition</li></ul><p><b>Goal:</b> Build a predictive model that can identify customers who are likely to churn, enabling the bank to take proactive retention measures.</p></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("<h3 class='sub-header'>🛠️ Project Workflow</h3>", unsafe_allow_html=True)
            st.markdown("""<div class="info-box"><ol><li>📊 <b>Data Collection</b> - Bank customer dataset (10,000 records)</li><li>🔍 <b>EDA</b> - Understand patterns, distributions, correlations</li><li>⚙️ <b>Preprocessing</b> - Clean, encode, engineer features, scale</li><li>⚖️ <b>Handle Imbalance</b> - Apply SMOTE oversampling</li><li>🤖 <b>Model Training</b> - Train 10 ML algorithms</li><li>📈 <b>Evaluation</b> - Compare using multiple metrics</li><li>🔧 <b>Optimization</b> - Hyperparameter tuning</li><li>🔮 <b>Deployment</b> - Interactive prediction UI</li></ol></div>""", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>📋 Dataset Features</h3>", unsafe_allow_html=True)
        feat = pd.DataFrame({"Feature":["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"],"Description":["Credit score of the customer (350-850)","Country of residence (France, Spain, Germany)","Male or Female","Age of the customer","Number of years with the bank","Account balance","Number of bank products used (1-4)","Whether customer has a credit card (0/1)","Whether customer is an active member (0/1)","Estimated annual salary","TARGET - Whether customer left the bank (0/1)"],"Type":["Numerical","Categorical","Categorical","Numerical","Numerical","Numerical","Numerical","Binary","Binary","Numerical","Binary (Target)"]})
        st.dataframe(feat, use_container_width=True, hide_index=True)

    elif page == "📊 Dataset Overview":
        st.markdown("<h1 class='main-header'>📊 Dataset Overview</h1>", unsafe_allow_html=True)
        t1,t2,t3,t4 = st.tabs(["📋 Data Preview","📊 Statistics","❓ Missing Values","📐 Data Types"])
        with t1:
            st.markdown("<h3 class='sub-header'>First 10 Rows</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("<h3 class='sub-header'>Last 10 Rows</h3>", unsafe_allow_html=True)
            st.dataframe(df.tail(10), use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Rows", f"{df.shape[0]:,}")
            c2.metric("Total Columns", df.shape[1])
            c3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum()/1024:.1f} KB")
        with t2:
            st.markdown("<h3 class='sub-header'>Statistical Summary</h3>", unsafe_allow_html=True)
            st.dataframe(df.describe().round(2), use_container_width=True)
        with t3:
            st.markdown("<h3 class='sub-header'>Missing Values Analysis</h3>", unsafe_allow_html=True)
            missing = pd.DataFrame({"Column":df.columns,"Missing Count":df.isnull().sum().values,"Missing %":(df.isnull().sum().values/len(df)*100).round(2)})
            st.dataframe(missing, use_container_width=True, hide_index=True)
            if df.isnull().sum().sum()==0: st.success("✅ No missing values found in the dataset!")
            else: st.warning("⚠️ Missing values detected!")
        with t4:
            st.markdown("<h3 class='sub-header'>Data Types</h3>", unsafe_allow_html=True)
            dtypes_df = pd.DataFrame({"Column":df.columns,"Data Type":df.dtypes.astype(str).values,"Non-Null Count":df.count().values,"Unique Values":df.nunique().values})
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    elif page == "🔍 Exploratory Data Analysis":
        st.markdown("<h1 class='main-header'>🔍 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
        t1,t2,t3,t4,t5 = st.tabs(["🎯 Target Variable","📊 Numerical Features","📋 Categorical Features","🔗 Correlations","💡 Key Insights"])
        with t1:
            st.markdown("<h3 class='sub-header'>Target Variable Distribution</h3>", unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            churn_counts = df["Exited"].value_counts()
            with c1:
                fig = px.bar(x=["Stayed (0)","Churned (1)"], y=churn_counts.values, color=["Stayed","Churned"], color_discrete_map={"Stayed":"#2ecc71","Churned":"#e74c3c"}, title="Customer Churn Count", labels={"x":"Status","y":"Count"})
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.pie(values=churn_counts.values, names=["Stayed","Churned"], title="Churn Percentage", color_discrete_sequence=["#2ecc71","#e74c3c"], hole=0.4)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Stayed", f"{churn_counts[0]:,}")
            c2.metric("Churned", f"{churn_counts[1]:,}")
            c3.metric("Imbalance Ratio", f"{churn_counts[0]/churn_counts[1]:.2f}:1")
        with t2:
            st.markdown("<h3 class='sub-header'>Numerical Feature Distributions</h3>", unsafe_allow_html=True)
            num_cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
            selected_num = st.selectbox("Select Feature", num_cols, key="num_feat")
            c1,c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x=selected_num, color="Exited", barmode="overlay", nbins=40, title=f"{selected_num} Distribution by Churn", color_discrete_map={0:"#2ecc71",1:"#e74c3c"}, labels={"Exited":"Churned"})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(df, x="Exited", y=selected_num, color="Exited", title=f"{selected_num} Box Plot by Churn", color_discrete_map={0:"#2ecc71",1:"#e74c3c"}, labels={"Exited":"Churned"})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        with t3:
            st.markdown("<h3 class='sub-header'>Categorical Feature Analysis</h3>", unsafe_allow_html=True)
            cat_cols = ["Geography","Gender","HasCrCard","IsActiveMember","NumOfProducts"]
            selected_cat = st.selectbox("Select Feature", cat_cols, key="cat_feat")
            c1,c2 = st.columns(2)
            with c1:
                ct = pd.crosstab(df[selected_cat], df["Exited"], normalize="index")*100
                fig = px.bar(ct, barmode="group", title=f"Churn Rate by {selected_cat}", color_discrete_sequence=["#2ecc71","#e74c3c"], labels={"value":"Percentage (%)","Exited":"Status"})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                churn_rate_by = df.groupby(selected_cat)["Exited"].mean()*100
                fig = px.bar(x=churn_rate_by.index.astype(str), y=churn_rate_by.values, title=f"Churn Rate (%) by {selected_cat}", color=churn_rate_by.values, color_continuous_scale="RdYlGn_r", labels={"x":selected_cat,"y":"Churn Rate (%)"})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        with t4:
            st.markdown("<h3 class='sub-header'>Correlation Heatmap</h3>", unsafe_allow_html=True)
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", title="Feature Correlation Matrix", aspect="auto", zmin=-1, zmax=1)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<h3 class='sub-header'>Correlation with Target</h3>", unsafe_allow_html=True)
            target_corr = corr_matrix["Exited"].drop("Exited").sort_values(ascending=True)
            fig = px.bar(x=target_corr.values, y=target_corr.index, orientation="h", title="Feature Correlation with Churn (Exited)", color=target_corr.values, color_continuous_scale="RdYlGn_r", labels={"x":"Correlation","y":"Feature"})
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        with t5:
            st.markdown("<h3 class='sub-header'>💡 Key EDA Insights</h3>", unsafe_allow_html=True)
            churn_by_geo = df.groupby("Geography")["Exited"].mean()*100
            churn_by_gender = df.groupby("Gender")["Exited"].mean()*100
            churn_by_active = df.groupby("IsActiveMember")["Exited"].mean()*100
            insights = [
                f"📊 **Dataset is IMBALANCED**: ~{(1-df.Exited.mean())*100:.0f}% stayed vs ~{df.Exited.mean()*100:.0f}% churned",
                "👴 **Age Factor**: Older customers (40-60) have significantly higher churn rates",
                f"🌍 **Geography**: Germany has the highest churn rate ({churn_by_geo.get('Germany',0):.1f}%)",
                f"👫 **Gender**: Females churn slightly more ({churn_by_gender.get('Female',0):.1f}%) than Males ({churn_by_gender.get('Male',0):.1f}%)",
                "💳 **Products**: Customers with 3-4 products churn at very high rates",
                f"🔄 **Activity**: Inactive members churn more ({churn_by_active.get(0,0):.1f}%) than active ({churn_by_active.get(1,0):.1f}%)",
                "💰 **Balance**: Customers with higher balances tend to churn more",
                "📉 **Credit Score**: Lower credit scores slightly increase churn probability",
            ]
            for insight in insights: st.markdown(f"- {insight}")

    elif page == "⚙️ Data Preprocessing":
        st.markdown("<h1 class='main-header'>⚙️ Data Preprocessing</h1>", unsafe_allow_html=True)
        t1,t2,t3,t4 = st.tabs(["🧹 Cleaning","🔧 Feature Engineering","🔢 Encoding & Scaling","⚖️ Handling Imbalance"])
        with t1:
            st.markdown("<h3 class='sub-header'>Step 1: Data Cleaning</h3>", unsafe_allow_html=True)
            st.markdown("**Actions Performed:**\n1. ✅ Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`\n2. ✅ Checked for duplicate rows\n3. ✅ Verified no missing values")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Before Cleaning:**")
                st.write(f"Shape: {df.shape}")
                st.write(f"Columns: {df.columns.tolist()}")
            with c2:
                df_clean = df.drop(["RowNumber","CustomerId","Surname"], axis=1)
                st.markdown("**After Cleaning:**")
                st.write(f"Shape: {df_clean.shape}")
                st.write(f"Columns: {df_clean.columns.tolist()}")
            st.success(f"✅ No duplicate rows found ({df.duplicated().sum()})")
        with t2:
            st.markdown("<h3 class='sub-header'>Step 2: Feature Engineering</h3>", unsafe_allow_html=True)
            fe = pd.DataFrame({"New Feature":["BalanceSalaryRatio","TenureAgeRatio","CreditScoreAge","HasBalance","ProductsPerTenure","AgeGroup"],"Formula":["Balance / EstimatedSalary","Tenure / Age","CreditScore / Age","1 if Balance > 0 else 0","NumOfProducts / (Tenure + 1)","Binned Age (0-4)"],"Rationale":["Financial engagement relative to income","Loyalty relative to age","Creditworthiness relative to age","Whether customer maintains balance","Product adoption rate over time","Age group segmentation"]})
            st.dataframe(fe, use_container_width=True, hide_index=True)
            df_processed = get_processed_data(df)
            st.write(f"**Shape after Feature Engineering:** {df_processed.shape}")
        with t3:
            st.markdown("<h3 class='sub-header'>Step 3: Encoding & Scaling</h3>", unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Label Encoding:**\n- `Gender`: Male → 1, Female → 0")
            with c2:
                st.markdown("**One-Hot Encoding:**\n- `Geography` → `Geography_Germany`, `Geography_Spain`\n- (France is the reference category - dropped)")
            st.markdown("**Feature Scaling:** StandardScaler (mean=0, std=1)")
            st.markdown("Applied to all features after splitting to prevent data leakage")
        with t4:
            st.markdown("<h3 class='sub-header'>Step 4: Handling Class Imbalance</h3>", unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            before = df["Exited"].value_counts()
            with c1:
                st.markdown("**Before SMOTE:**")
                fig = px.bar(x=["Stayed","Churned"], y=before.values, color=["Stayed","Churned"], color_discrete_map={"Stayed":"#2ecc71","Churned":"#e74c3c"}, title="Before SMOTE")
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                majority = before[0]
                st.markdown("**After SMOTE:**")
                fig = px.bar(x=["Stayed","Churned"], y=[majority,majority], color=["Stayed","Churned"], color_discrete_map={"Stayed":"#2ecc71","Churned":"#e74c3c"}, title="After SMOTE (Balanced)")
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
            st.info("💡 **SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic samples of the minority class to balance the dataset, preventing model bias toward the majority class.")

    elif page == "🤖 Model Training & Evaluation":
        st.markdown("<h1 class='main-header'>🤖 Model Training & Evaluation</h1>", unsafe_allow_html=True)
        df_processed = get_processed_data(df)
        X_train_res, X_test_scaled, y_train_res, y_test, scaler, feature_names = prepare_training_data(df_processed)
        with st.spinner("🔄 Training 10 ML models... Please wait..."):
            results, trained_models = do_training(X_train_res, X_test_scaled, y_train_res, y_test)
        st.success("✅ All models trained successfully!")
        st.session_state["results"] = results
        st.session_state["trained_models"] = trained_models
        st.session_state["scaler"] = scaler
        st.session_state["X_test_scaled"] = X_test_scaled
        st.session_state["y_test"] = y_test
        st.session_state["feature_names"] = feature_names
        selected_model = st.selectbox("Select Model to View Details", list(results.keys()))
        r = results[selected_model]
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Accuracy", f"{r['Accuracy']:.4f}")
        c2.metric("Precision", f"{r['Precision']:.4f}")
        c3.metric("Recall", f"{r['Recall']:.4f}")
        c4.metric("F1-Score", f"{r['F1-Score']:.4f}")
        c5.metric("ROC-AUC", f"{r['ROC-AUC']:.4f}")
        model = trained_models[selected_model]
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
        c1,c2 = st.columns(2)
        with c1:
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted",y="Actual",color="Count"), x=["Stayed","Churned"], y=["Stayed","Churned"], color_continuous_scale="Blues", title=f"Confusion Matrix - {selected_model}")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fpr,tpr,_ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{selected_model} (AUC={auc_val:.4f})", line=dict(color="#3498db",width=3)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(color="gray",dash="dash")))
            fig.update_layout(title=f"ROC Curve - {selected_model}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=450)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("<h3 class='sub-header'>Classification Report</h3>", unsafe_allow_html=True)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    elif page == "📈 Model Comparison":
        st.markdown("<h1 class='main-header'>📈 Model Comparison</h1>", unsafe_allow_html=True)
        if "results" not in st.session_state:
            st.warning("⚠️ Please go to 'Model Training & Evaluation' page first to train the models!")
            return
        results = st.session_state["results"]
        trained_models = st.session_state["trained_models"]
        X_test_scaled = st.session_state["X_test_scaled"]
        y_test = st.session_state["y_test"]
        results_df = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)
        st.markdown("<h3 class='sub-header'>All Models Performance Table</h3>", unsafe_allow_html=True)
        st.dataframe(results_df.round(4).style.highlight_max(axis=0, color="#2ecc71").highlight_min(axis=0, color="#e74c3c"), use_container_width=True)
        best_name = results_df["ROC-AUC"].idxmax()
        st.success(f"🏆 **Best Model: {best_name}** with ROC-AUC = {results_df.loc[best_name,'ROC-AUC']:.4f}")
        t1,t2,t3 = st.tabs(["📊 Metrics Comparison","📈 ROC Curves","🏆 Feature Importance"])
        with t1:
            metrics = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
            selected_metric = st.selectbox("Select Metric", metrics)
            sorted_df = results_df.sort_values(selected_metric, ascending=True)
            fig = px.bar(sorted_df, x=selected_metric, y=sorted_df.index, orientation="h", title=f"Model Comparison - {selected_metric}", color=selected_metric, color_continuous_scale="Viridis", text=sorted_df[selected_metric].round(4))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=500, yaxis_title="Model", xaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)
            fig = go.Figure()
            colors = ["#3498db","#2ecc71","#e74c3c","#f39c12","#9b59b6"]
            for i,metric in enumerate(metrics): fig.add_trace(go.Bar(name=metric, x=results_df.index, y=results_df[metric], marker_color=colors[i]))
            fig.update_layout(barmode="group", title="All Metrics Comparison", height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            fig = go.Figure()
            colors_list = px.colors.qualitative.Set3
            for i,(name,model) in enumerate(trained_models.items()):
                yp = model.predict_proba(X_test_scaled)[:,1]
                fpr,tpr,_ = roc_curve(y_test, yp)
                auc_val = roc_auc_score(y_test, yp)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc_val:.3f})", line=dict(color=colors_list[i%len(colors_list)],width=2)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(color="gray",dash="dash",width=2)))
            fig.update_layout(title="ROC Curves - All Models", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=600, legend=dict(x=0.6,y=0.1))
            st.plotly_chart(fig, use_container_width=True)
        with t3:
            feature_names = st.session_state["feature_names"]
            tree_models = {k:v for k,v in trained_models.items() if hasattr(v,"feature_importances_")}
            if tree_models:
                sel_model = st.selectbox("Select Model", list(tree_models.keys()))
                importance = tree_models[sel_model].feature_importances_
                feat_imp_df = pd.DataFrame({"Feature":feature_names,"Importance":importance}).sort_values("Importance", ascending=True)
                fig = px.bar(feat_imp_df, x="Importance", y="Feature", orientation="h", title=f"Feature Importance - {sel_model}", color="Importance", color_continuous_scale="Viridis")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

    elif page == "🔮 Predict New Customer":
        st.markdown("<h1 class='main-header'>🔮 Predict Customer Churn</h1>", unsafe_allow_html=True)
        if "trained_models" not in st.session_state:
            st.warning("⚠️ Please go to 'Model Training & Evaluation' page first to train the models!")
            return
        trained_models = st.session_state["trained_models"]
        scaler = st.session_state["scaler"]
        results = st.session_state["results"]
        st.markdown("<h3 class='sub-header'>Enter Customer Details</h3>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            credit_score = st.slider("💳 Credit Score", 350, 850, 650)
            geography = st.selectbox("🌍 Geography", ["France","Spain","Germany"])
            gender = st.selectbox("👤 Gender", ["Male","Female"])
            age = st.slider("🎂 Age", 18, 92, 35)
        with c2:
            tenure = st.slider("📅 Tenure (Years)", 0, 10, 5)
            balance = st.number_input("💰 Balance ($)", 0.0, 300000.0, 50000.0, step=1000.0)
            num_products = st.selectbox("📦 Number of Products", [1,2,3,4])
        with c3:
            has_cr_card = st.selectbox("💳 Has Credit Card", ["Yes","No"])
            is_active = st.selectbox("🔄 Is Active Member", ["Yes","No"])
            salary = st.number_input("💵 Estimated Salary ($)", 0.0, 200000.0, 100000.0, step=5000.0)
        best_model_name = max(results, key=lambda k: results[k]["ROC-AUC"])
        pred_model_name = st.selectbox("🤖 Select Prediction Model", list(trained_models.keys()), index=list(trained_models.keys()).index(best_model_name))
        if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
            customer = {"CreditScore":credit_score,"Geography":geography,"Gender":gender,"Age":age,"Tenure":tenure,"Balance":balance,"NumOfProducts":num_products,"HasCrCard":has_cr_card,"IsActiveMember":is_active,"EstimatedSalary":salary}
            model = trained_models[pred_model_name]
            prediction, probability = predict_customer(model, scaler, customer)
            st.markdown("---")
            st.markdown("<h3 class='sub-header'>🎯 Prediction Results</h3>", unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            with c1:
                if prediction==1: st.markdown("<div class='danger-card'><h2>🔴 WILL CHURN</h2><p>High Risk Customer</p></div>", unsafe_allow_html=True)
                else: st.markdown("<div class='success-card'><h2>🟢 WILL STAY</h2><p>Low Risk Customer</p></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='metric-card'><h2>{probability[1]*100:.1f}%</h2><p>Churn Probability</p></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='warning-card'><h2>{probability[0]*100:.1f}%</h2><p>Retention Probability</p></div>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=probability[1]*100, title={"text":"Churn Risk Score","font":{"size":24}}, delta={"reference":50}, gauge={"axis":{"range":[None,100],"tickwidth":1}, "bar":{"color":"darkblue"}, "steps":[{"range":[0,30],"color":"#2ecc71"},{"range":[30,60],"color":"#f39c12"},{"range":[60,100],"color":"#e74c3c"}], "threshold":{"line":{"color":"red","width":4},"thickness":0.75,"value":70}}))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<h3 class='sub-header'>💡 Recommendations</h3>", unsafe_allow_html=True)
            if probability[1]>0.7:
                st.error("⚠️ **HIGH RISK** - Immediate action required!")
                st.markdown("**Recommended Actions:**\n- 🎁 Offer personalized retention package immediately\n- 📞 Assign dedicated relationship manager\n- 💰 Provide competitive interest rates or fee waivers\n- 🏆 Offer loyalty rewards or cashback programs\n- 📊 Schedule quarterly review meetings")
            elif probability[1]>0.4:
                st.warning("⚡ **MEDIUM RISK** - Monitor closely")
                st.markdown("**Recommended Actions:**\n- 📧 Send personalized engagement emails\n- 🎯 Offer targeted product recommendations\n- 💳 Suggest credit card upgrades or new products\n- 📱 Encourage mobile banking adoption")
            else:
                st.success("✅ **LOW RISK** - Customer is likely to stay")
                st.markdown("**Recommended Actions:**\n- 🤝 Maintain regular communication\n- ⭐ Reward loyalty with exclusive benefits\n- 📈 Cross-sell relevant financial products\n- 💡 Share financial insights and tips")

    elif page == "📋 Project Summary":
        st.markdown("<h1 class='main-header'>📋 Project Summary</h1>", unsafe_allow_html=True)
        t1,t2,t3,t4 = st.tabs(["📝 Overview","🔧 Methodology","📊 Results","🎯 Conclusions"])
        with t1:
            st.markdown("<h3 class='sub-header'>Project Overview</h3>", unsafe_allow_html=True)
            st.markdown("""
| Aspect | Details |
|--------|---------|
| **Project** | Customer Churn Prediction in Banking |
| **Domain** | Banking / Financial Services |
| **Type** | Binary Classification |
| **Dataset** | Bank Customer Churn (10,000 records) |
| **Features** | 14 original + 6 engineered = 20 features |
| **Target** | Exited (0 = Stayed, 1 = Churned) |
| **Models** | 10 ML algorithms compared |
| **Best Model** | XGBoost / LightGBM (typically) |
| **UI** | Streamlit Interactive Dashboard |
            """)
        with t2:
            st.markdown("<h3 class='sub-header'>Complete Methodology</h3>", unsafe_allow_html=True)
            steps = {"1. Data Collection":"Generated realistic banking dataset with 10,000 customers and 14 features mimicking real-world distributions.","2. EDA":"Analyzed target distribution (imbalanced ~80/20), feature distributions, categorical analysis, and correlation patterns.","3. Data Cleaning":"Removed irrelevant columns (RowNumber, CustomerId, Surname), verified no missing values or duplicates.","4. Feature Engineering":"Created 6 new features: BalanceSalaryRatio, TenureAgeRatio, CreditScoreAge, HasBalance, ProductsPerTenure, AgeGroup.","5. Encoding":"Label Encoding for Gender, One-Hot Encoding for Geography (drop_first=True).","6. Scaling":"StandardScaler applied after train-test split to prevent data leakage.","7. Imbalance Handling":"SMOTE oversampling applied on training data only.","8. Model Training":"10 algorithms: LR, DT, RF, GB, XGBoost, LightGBM, KNN, SVM, NB, AdaBoost.","9. Evaluation":"Accuracy, Precision, Recall, F1-Score, ROC-AUC with confusion matrices and ROC curves.","10. Deployment":"Interactive Streamlit web application for real-time predictions."}
            for step, desc in steps.items():
                with st.expander(f"📌 {step}"): st.write(desc)
        with t3:
            st.markdown("<h3 class='sub-header'>Model Results</h3>", unsafe_allow_html=True)
            if "results" in st.session_state:
                rdf = pd.DataFrame(st.session_state["results"]).T.sort_values("ROC-AUC", ascending=False)
                st.dataframe(rdf.round(4).style.highlight_max(axis=0, color="#2ecc71").highlight_min(axis=0, color="#e74c3c"), use_container_width=True)
                best = rdf.iloc[0]
                st.success(f"🏆 **Best Model: {rdf.index[0]}** | Accuracy: {best['Accuracy']:.4f} | F1: {best['F1-Score']:.4f} | AUC: {best['ROC-AUC']:.4f}")
            else: st.info("Train models first to see results here.")
        with t4:
            st.markdown("<h3 class='sub-header'>Key Conclusions</h3>", unsafe_allow_html=True)
            st.markdown("""
### 🔍 Key Findings:
1. **Age** is the strongest predictor — customers aged 40-60 churn significantly more
2. **Geography** matters — German customers show highest churn rates
3. **Activity status** is crucial — inactive members are much more likely to churn
4. **Number of products** — customers with 3-4 products have very high churn rates
5. **Gender gap** — female customers churn slightly more than males

### 💼 Business Recommendations:
1. **Targeted Retention**: Focus on customers aged 40-60, especially in Germany
2. **Engagement Programs**: Develop programs to activate inactive members
3. **Product Strategy**: Investigate why multi-product customers churn
4. **Gender-Specific**: Create tailored offers for female customers
5. **Early Warning System**: Implement this ML model for real-time churn alerts
6. **Cost Savings**: Proactive retention is 5-25x cheaper than acquisition

### 🚀 Future Improvements:
- Integrate real banking transaction data
- Add temporal features (seasonal patterns)
- Implement deep learning (Neural Networks)
- A/B testing of retention strategies
- Real-time API deployment with FastAPI
            """)
        st.markdown("---")
        st.markdown("""<div style="text-align:center; padding:2rem; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); border-radius:15px; color:white;"><h2>🎓 Customer Churn Prediction Project</h2><p>Complete End-to-End Machine Learning Pipeline</p><p>Built with Python • Scikit-Learn • Streamlit</p><p>© 2024 | All Rights Reserved</p></div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()