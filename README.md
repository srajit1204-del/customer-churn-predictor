# Customer Churn Predictor

## Overview
This project predicts customer churn using Machine Learning on the Telco Customer Churn dataset.

## Features
- Data preprocessing and feature engineering
- Logistic Regression model training
- Evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Streamlit web application for churn prediction

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- SQLite

## Project Structure
- `data/` → dataset files
- `notebooks/` → experimentation notebooks
- `models/` → trained models
- `src/` → source code
- `app/` → Streamlit app

## How to Run

```bash
pip install -r requirements.txt
streamlit run app/app.py