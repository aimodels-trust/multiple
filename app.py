# -*- coding: utf-8 -*-
"""app.py"""

import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib

def load_model(model_path):
    return joblib.load(model_path)

def preprocess_data(df, target_column):
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in uploaded data.")
        return None, None

    X = df.drop(columns=[target_column], errors='ignore')
    return X, df[target_column] if target_column in df.columns else None

def explain_predictions(model, X):
    try:
        model_type = str(type(model)).lower()

        if "xgboost" in model_type or "randomforest" in model_type:
            explainer = shap.TreeExplainer(model)
        elif "linear" in model_type:
            explainer = shap.LinearExplainer(model, X)
        else:
            explainer = shap.Explainer(model, X)

        shap_values = explainer(X[:100])  # Limit to 100 samples for efficiency
        return shap_values

    except Exception as e:
        st.error(f"SHAP explainability error: {e}")
        return None

def main():
    st.title("Trust & Transparency in Business Systems using Explainable AI")

    # Select domain
    domain = st.selectbox("Select Domain", ["Finance", "Healthcare", "HR & Recruitment", "Marketing"])

    # Select sub-category based on domain
    sub_categories = {
        "Finance": ["Credit Scoring", "Fraud Detection"],
        "Healthcare": ["Medical Diagnosis", "Patient Risk Assessment"],
        "HR & Recruitment": ["Candidate Screening", "Employee Performance Predictions"],
        "Marketing": ["Customer Segmentation", "Recommendation Systems"]
    }
    sub_category = st.selectbox("Select Application", sub_categories.get(domain, ["Coming Soon"]))

    # Upload dataset
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.write(df.head())

        # User selects classification column
        target_column = st.selectbox("Select Classification Column", df.columns)

        # Load corresponding model dynamically
        model_mapping = {
            "Fraud Detection": "models/fraud_detection.joblib",
            "Credit Scoring": "models/credit_scoring.joblib",
            "Medical Diagnosis": "models/medical_diagnosis.pkl",
            "Patient Risk Assessment": "models/patient_risk.pkl",
            "Candidate Screening": "models/candidate_screening.pkl",
            "Employee Performance Predictions": "models/employee_performance.pkl",
            "Customer Segmentation": "models/customer_segmentation.pkl",
            "Recommendation Systems": "models/recommendation_system.pkl"
        }

        model_path = model_mapping.get(sub_category)

        if model_path:
            model = load_model(model_path)

            # Preprocess data
            X, y = preprocess_data(df, target_column)
            if X is not None:
                # Make predictions
                predictions = model.predict(X)
                df["Predictions"] = predictions
                st.write("### Prediction Results:")
                st.write(df[[target_column, "Predictions"]])

                # Explain with SHAP
                st.write("### Explainability (SHAP Values):")
                shap_values = explain_predictions(model, X)

                if shap_values is not None:
                    try:
                        # SHAP Force Plot
                        shap.initjs()
                        fig, ax = plt.subplots()
                        shap.force_plot(
                            shap_values.base_values[0], 
                            shap_values.values[0, :], 
                            X.iloc[0, :], 
                            matplotlib=True
                        )
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"SHAP force plot error: {e}")

                else:
                    st.warning("SHAP explainability failed.")
        else:
            st.warning("Model for the selected application is not available yet.")

if __name__ == "__main__":
    main()
