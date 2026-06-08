# Customer Churn Predictor

Live Demo:
https://customer-churn-predictor-ngns4dejzt8ghz9q6lrnok.streamlit.app/

---

## About the Project

This project predicts whether a telecom customer is likely to churn based on customer details such as tenure, monthly charges, internet service, payment method, and support history.

The model is deployed using Streamlit, where users can enter customer information and get churn probability along with a simple explanation of the prediction.

---

## Features

* Customer churn prediction
* Interactive Streamlit interface
* SHAP-based prediction explanation
* Comparison of different ML models
* Data preprocessing and feature engineering

---

## Models Used

* Logistic Regression
* Random Forest
* XGBoost

XGBoost was used as the final model for prediction.

---

## Model Performance

| Model               | ROC-AUC | F1 Score |
| ------------------- | ------- | -------- |
| Logistic Regression | 0.8460  | 0.59     |
| Random Forest       | 0.8254  | 0.64     |
| XGBoost             | 0.8376  | 0.61     |

XGBoost was selected as the final model for deployment.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Matplotlib

---

## Project Structure

```bash
Customer-Churn-Predictor/
│
├── app.py
├── models/
│   ├── xgb_model.pkl
│   └── shap_explainer.pkl
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_shap.ipynb
│
├── data/
├── requirements.txt
└── README.md
```

---

## Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Output

The application shows:

* Churn probability
* Risk level
* SHAP explanation for prediction
* Model comparison metrics
