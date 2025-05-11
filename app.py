import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px  # ✅ Needed for histogram

from model import train_model, predict_attrition, evaluate_model
from utils import display_kpis, plot_confusion_matrix, feature_importance_plot, plot_calibration_curve

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
st.title("🔍 HR Attrition Risk Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("employee_data.csv")

df = load_data()

# Sidebar filter for Education
st.sidebar.header("📊 Filters")
education_levels = st.sidebar.multiselect("Select Education Level(s):",
                                          options=sorted(df["Education"].unique()),
                                          default=sorted(df["Education"].unique()))
filtered_df = df[df["Education"].isin(education_levels)]

# Train model
model, X_train, y_train, X_test, y_test = train_model(filtered_df)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
metrics = evaluate_model(y_test, y_pred, y_proba)

# KPI Section
st.subheader("📈 Key Performance Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
col2.metric("Recall", f"{metrics['recall']*100:.2f}%")
col3.metric("Precision", f"{metrics['precision']*100:.2f}%")

# Gauge chart
st.subheader("📉 Predicted Attrition Recall")
gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=metrics['recall']*100,
    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "crimson"}},
    title={'text': "Recall Score (%)"}
))
st.plotly_chart(gauge, use_container_width=True)

# Attrition Histogram
st.subheader("📊 Attrition by Job Level")
chart = px.histogram(filtered_df, x="JobLevel", color="Attrition", barmode="group")
st.plotly_chart(chart, use_container_width=True)

# Top at-risk employees
st.subheader("🚨 Top At-Risk Employees")
X_filtered = filtered_df.drop("Attrition", axis=1)
filtered_df["RiskScore"] = model.predict_proba(X_filtered)[:, 1]
top_risk = filtered_df.sort_values(by="RiskScore", ascending=False).head(10)
st.dataframe(top_risk[["Age", "JobLevel", "MonthlyIncome", "RiskScore"]].round(2))

# Actionable insights
st.subheader("✅ Action Recommendations")
avg_risk = top_risk["RiskScore"].mean()
if avg_risk > 0.5:
    st.warning("⚠ High average attrition risk detected. Recommend initiating employee engagement or retention programs.")

# Feature importance
st.subheader("🔬 Feature Importance")
feature_importance_plot(model, X_train)

# Calibration & Confusion Matrix
st.subheader("📉 Model Calibration (ROC Curve)")
plot_calibration_curve(model, X_test, y_test)

st.subheader("📋 Confusion Matrix")
plot_confusion_matrix(metrics["confusion_matrix"])

st.subheader("📌 Detailed Evaluation Metrics")
display_kpis(metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"], metrics["roc_auc"])

# Footer
st.markdown("---")
st.caption("Model Version: v1.0 | Trained on: employee_data.csv")
