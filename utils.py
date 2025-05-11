import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

def display_kpis(acc, prec, rec, f1, auc):
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**Precision:** {prec:.2f}")
    st.write(f"**Recall:** {rec:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")
    st.write(f"**ROC AUC Score:** {auc:.2f}")

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

def feature_importance_plot(model, features):
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    plt.figure(figsize=(10, 5))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[sorted_idx], align="center")
    plt.xticks(range(len(importances)), [features.columns[i] for i in sorted_idx], rotation=45)
    st.pyplot(plt.gcf())

def plot_calibration_curve(model, X_test, y_test):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)
