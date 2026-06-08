import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"

model    = pickle.load(open(path + "models/xgb_model.pkl",      "rb"))
explainer = pickle.load(open(path + "models/shap_explainer.pkl", "rb"))
scaler   = pickle.load(open(path + "data/scaler.pkl",           "rb"))
X_test   = pd.read_csv(path + "data/X_test.csv")

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Predictor")
st.markdown("Enter customer details in the sidebar and click **Predict**.")

# ── Sidebar inputs ────────────────────────────────────────
st.sidebar.header("Customer Details")

tenure          = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
contract        = st.sidebar.selectbox("Contract Type",
                    ["Month-to-month", "One year", "Two year"])
internet        = st.sidebar.selectbox("Internet Service",
                    ["Fiber optic", "DSL", "No"])
tech_support    = st.sidebar.selectbox("Tech Support",
                    ["No", "Yes", "No internet service"])
payment         = st.sidebar.selectbox("Payment Method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"])
senior          = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner         = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents      = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

predict_btn = st.sidebar.button("Predict Churn Risk")

# ── Prediction logic ──────────────────────────────────────
if predict_btn:

    # Build input row matching X_test columns
    input_dict = {col: [0] for col in X_test.columns}

    # Fill numerical features
    input_dict["tenure"]         = [tenure]
    input_dict["MonthlyCharges"] = [monthly_charges]
    input_dict["SeniorCitizen"]  = [1 if senior == "Yes" else 0]
    input_dict["Partner"]        = [1 if partner == "Yes" else 0]
    input_dict["Dependents"]     = [1 if dependents == "Yes" else 0]

    # Engineered features
    total_charges = monthly_charges * tenure
    input_dict["TotalCharges"]        = [total_charges]
    input_dict["avg_monthly_spend"]   = [total_charges / (tenure + 1)]
    input_dict["is_new_customer"]     = [1 if tenure <= 12 else 0]
    input_dict["high_value_customer"] = [1 if monthly_charges > 79 else 0]

    # One-hot encoded features
    if contract == "One year":
        input_dict["Contract_One year"] = [1]
    elif contract == "Two year":
        input_dict["Contract_Two year"] = [1]

    if internet == "Fiber optic":
        input_dict["InternetService_Fiber optic"] = [1]
    elif internet == "No":
        input_dict["InternetService_No"] = [1]

    if tech_support == "Yes":
        input_dict["TechSupport_Yes"] = [1]
    elif tech_support == "No internet service":
        input_dict["TechSupport_No internet service"] = [1]

    if payment == "Credit card (automatic)":
        input_dict["PaymentMethod_Credit card (automatic)"] = [1]
    elif payment == "Electronic check":
        input_dict["PaymentMethod_Electronic check"] = [1]
    elif payment == "Mailed check":
        input_dict["PaymentMethod_Mailed check"] = [1]

    input_df = pd.DataFrame(input_dict)

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prob  = model.predict_proba(input_scaled)[0][1]
    risk  = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
    color = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"

    # ── Show metrics ──────────────────────────────────────
    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{prob:.1%}")
    col2.metric("Risk Level", f"{color} {risk}")

    st.divider()

    # ── SHAP explanation ──────────────────────────────────
    st.subheader("Why this prediction?")
    shap_vals = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns.tolist()
        ),
        max_display=10,
        show=False
    )
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Model summary ─────────────────────────────────────
    st.subheader("Model Comparison")
    results = pd.DataFrame({
        "Model":       ["Logistic Regression", "Random Forest", "XGBoost"],
        "ROC-AUC":     [0.8460, 0.8254, 0.8376],
        "F1 (churn)":  [0.59, 0.64, 0.61],
        "Selected":    ["", "", "✅"]
    })
    st.dataframe(results, hide_index=True)