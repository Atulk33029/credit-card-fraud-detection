import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import FraudPredictor

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

@st.cache_resource
def load_predictor():
    return FraudPredictor(model_path='models/model.json', config_path='models/config.json')

@st.cache_data
def load_metrics():
    with open('models/metrics.json') as f:
        return json.load(f)

@st.cache_data
def load_sample_data():
    return pd.read_csv('models/scored_transactions.csv')

predictor = load_predictor()
metrics = load_metrics()

st.title("💳 Credit Card Fraud Detection")
st.caption("XGBoost model trained on 284,807 transactions (Kaggle) — PR-AUC 0.856, 85% precision / 84% recall on the fraud class")

tab1, tab2, tab3 = st.tabs(["🔍 Try a Prediction", "📊 Model Performance", "📈 Dataset Insights"])

# ---------------- TAB 1: Live prediction ----------------
with tab1:
    st.subheader("Score a transaction")
    st.write("This dataset's `V1`-`V28` features are PCA-anonymized, so there's no real-world meaning to enter manually. "
             "Use a sample from the test set below, or set custom values to see how the model reacts.")

    sample_df = load_sample_data()

    mode = st.radio("Input mode", ["Load a real test-set example", "Custom values"], horizontal=True)

    if mode == "Load a real test-set example":
        col1, col2 = st.columns(2)
        with col1:
            show_fraud_only = st.checkbox("Show only actual fraud cases", value=True)
        filtered = sample_df[sample_df['y_true'] == 1] if show_fraud_only else sample_df
        idx = st.selectbox("Pick a transaction (by row index)", filtered.index[:50])
        row = filtered.loc[idx]
        st.write(f"**Actual label:** {'Fraud' if row['y_true']==1 else 'Legit'} | "
                 f"**Amount:** ${row['Amount']:.2f} | **Hour:** {int(row['Hour'])}")
        st.info("Note: this app only stores the scored outputs, not the original V-features, "
                "so this demo re-uses the model's own stored probability for real test rows.")
        proba = row['y_proba']
        is_fraud = proba >= predictor.threshold
    else:
        st.write("Set each PCA feature (defaults to 0 = 'average' transaction):")
        cols = st.columns(4)
        v_features = {}
        for i in range(1, 29):
            with cols[(i-1) % 4]:
                v_features[f'V{i}'] = st.slider(f'V{i}', -10.0, 10.0, 0.0, 0.1, key=f'v{i}')
        amount = st.number_input("Amount ($)", min_value=0.0, value=50.0, step=1.0)
        hour = st.slider("Hour of day", 0, 23, 14)

        result = predictor.predict(v_features, amount, hour)
        proba = result['fraud_probability']
        is_fraud = result['is_fraud']

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability", f"{proba*100:.2f}%")
    c2.metric("Decision Threshold", f"{predictor.threshold*100:.2f}%")
    c3.metric("Verdict", "🚨 FRAUD" if is_fraud else "✅ Legit")

# ---------------- TAB 2: Model performance ----------------
with tab2:
    st.subheader("Model comparison")
    comp_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'PR-AUC': [metrics['logreg']['pr_auc'], metrics['random_forest']['pr_auc'], metrics['xgboost']['pr_auc']],
        'ROC-AUC': [metrics['logreg']['roc_auc'], metrics['random_forest']['roc_auc'], metrics['xgboost']['roc_auc']],
    })
    st.dataframe(comp_df, use_container_width=True)
    st.caption("PR-AUC is the metric that matters here — with fraud at 0.17% of transactions, "
               "accuracy alone is misleading (predicting 'no fraud' always would score 99.8%).")

    col1, col2 = st.columns(2)
    with col1:
        st.image("reports/figures/pr_curve.png", caption="Precision-Recall Curve")
    with col2:
        st.image("reports/figures/confusion_matrix.png", caption="Confusion Matrix (XGBoost, test set)")
    st.image("reports/figures/model_comparison.png", caption="PR-AUC by model")

# ---------------- TAB 3: Dataset insights ----------------
with tab3:
    st.subheader("Fraud patterns in the test set")
    fraud_by_hour = sample_df.groupby('Hour')['y_true'].agg(['sum', 'count'])
    fraud_by_hour['rate'] = fraud_by_hour['sum'] / fraud_by_hour['count']
    st.bar_chart(fraud_by_hour['rate'])
    st.caption("Fraud rate by hour of day")

    st.subheader("Amount distribution: fraud vs legit")
    st.write(sample_df.groupby('y_true')['Amount'].describe())