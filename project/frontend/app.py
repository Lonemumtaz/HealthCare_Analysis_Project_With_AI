"""
Healthcare Test Results Prediction — Streamlit Frontend
========================================================
Loads the trained pipeline and provides an interactive UI.
"""

import os
import sys
import json
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')

# ── Resolve paths relative to this file ─────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'trained_model.pkl')
META_PATH  = os.path.join(BASE_DIR, 'model', 'feature_meta.json')
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'dataset.csv')
FIGURES    = os.path.join(BASE_DIR, 'outputs', 'figures')
METRICS    = os.path.join(BASE_DIR, 'outputs', 'metrics')

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Test Results Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model & metadata ────────────────────────────────────────────
@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact['pipeline'], artifact['label_encoder'], artifact['best_model_name']

@st.cache_data
def load_meta():
    with open(META_PATH, 'r') as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics():
    path = os.path.join(METRICS, 'best_model_metrics.json')
    with open(path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_comparison():
    path = os.path.join(METRICS, 'model_comparison.csv')
    return pd.read_csv(path)

try:
    pipeline, le, best_model_name = load_model()
    meta = load_meta()
    df_raw = load_data()
    best_metrics = load_metrics()
    comparison_df = load_comparison()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/hospital-3.png",
    width=80
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Dataset Overview", "🔮 Make Prediction", "📈 Model Performance", "📉 Visualisations"]
)

# ════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏥 Healthcare Test Results Prediction")
    st.markdown("""
    ## Project Overview

    This application predicts a patient's **medical test result** (Normal, Inconclusive, or Abnormal)
    based on demographic and clinical features extracted from healthcare admission records.

    ### Problem Type
    **Multiclass Classification** — 3 target classes: `Normal`, `Inconclusive`, `Abnormal`

    ### Dataset
    | Property | Value |
    |---|---|
    | Source | Healthcare Dataset (Kaggle) |
    | Rows | 55,500 |
    | Columns | 15 |
    | Target Column | Test Results |
    | Problem Type | Multiclass Classification |

    ### Best Model
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", best_model_name)
    col2.metric("Accuracy", f"{best_metrics['accuracy']:.2%}")
    col3.metric("F1-Score", f"{best_metrics['f1_score']:.2%}")
    col4.metric("ROC-AUC", f"{float(best_metrics['roc_auc']):.4f}")

    st.markdown("""
    ### Note on Model Performance
    > This dataset (a popular synthetic Kaggle healthcare dataset) has **near-uniformly distributed**
    > test result labels with no strong predictive signals in the available features.
    > The best achievable accuracy is approximately **43%** — significantly better than random
    > guessing (33.3% for 3 balanced classes), but limited by the synthetic nature of the data.
    > This is consistent with published analyses of this specific dataset.

    ### How to Use
    - Use **Make Prediction** to input patient details and get a predicted test result.
    - Use **Dataset Overview** to explore the data.
    - Use **Model Performance** to review model metrics and comparisons.
    - Use **Visualisations** to view EDA charts.
    """)

# ════════════════════════════════════════════════════════════════════
# PAGE: DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df_raw):,}")
    col2.metric("Features", df_raw.shape[1] - 1)
    col3.metric("Target Classes", 3)

    st.subheader("First 10 Rows")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df_raw.describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Distribution")
        vc = df_raw['Test Results'].value_counts()
        st.bar_chart(vc)

    with col2:
        st.subheader("Data Quality")
        quality = pd.DataFrame({
            'Missing Values': df_raw.isnull().sum(),
            'Unique Values': df_raw.nunique(),
            'Data Type': df_raw.dtypes.astype(str)
        })
        st.dataframe(quality, use_container_width=True)

    st.subheader("Feature Description")
    feature_desc = pd.DataFrame({
        'Column': df_raw.columns,
        'Description': [
            'Patient full name', 'Patient age (years)', 'Patient gender',
            'Patient blood type', 'Primary medical condition',
            'Hospital admission date', 'Attending doctor name',
            'Hospital name', 'Insurance provider', 'Billing amount ($)',
            'Assigned room number', 'Type of admission',
            'Hospital discharge date', 'Prescribed medication',
            'Medical test result (TARGET)'
        ]
    })
    st.dataframe(feature_desc, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════
# PAGE: MAKE PREDICTION
# ════════════════════════════════════════════════════════════════════
elif page == "🔮 Make Prediction":
    st.title("🔮 Predict Patient Test Result")
    st.markdown("Enter patient information below and click **Predict** to get a test result prediction.")

    # Get unique values from meta
    cat_vals = meta['cat_unique_values']

    with st.form("prediction_form"):
        st.subheader("Patient Information")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
            gender = st.selectbox("Gender", options=sorted(cat_vals.get('Gender', ['Male', 'Female'])))
            blood_type = st.selectbox("Blood Type", options=sorted(cat_vals.get('Blood Type', ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'])))
            medical_condition = st.selectbox("Medical Condition", options=sorted(cat_vals.get('Medical Condition', ['Cancer', 'Obesity', 'Diabetes', 'Asthma', 'Hypertension', 'Arthritis'])))

        with col2:
            insurance_provider = st.selectbox("Insurance Provider", options=sorted(cat_vals.get('Insurance Provider', ['Aetna', 'Blue Cross', 'Cigna', 'Medicare', 'UnitedHealthcare'])))
            billing_amount = st.number_input("Billing Amount ($)", min_value=0.0, max_value=200000.0, value=25000.0, step=100.0)
            room_number = st.number_input("Room Number", min_value=100, max_value=500, value=300, step=1)
            admission_type = st.selectbox("Admission Type", options=sorted(cat_vals.get('Admission Type', ['Elective', 'Emergency', 'Urgent'])))

        st.subheader("Additional Details")
        col3, col4 = st.columns(2)
        with col3:
            medication = st.selectbox("Medication", options=sorted(cat_vals.get('Medication', ['Aspirin', 'Ibuprofen', 'Lipitor', 'Paracetamol', 'Penicillin'])))
            length_of_stay = st.slider("Length of Stay (days)", min_value=1, max_value=30, value=10)

        with col4:
            admission_year = st.selectbox("Admission Year", options=list(range(2019, 2026)), index=3)
            admission_month = st.selectbox("Admission Month", options=list(range(1, 13)), index=0)
            admission_dayofweek = st.selectbox("Day of Week (0=Mon, 6=Sun)", options=list(range(0, 7)), index=0)

        submitted = st.form_submit_button("🔮 Predict Test Result", use_container_width=True)

    if submitted:
        # Build input DataFrame matching training feature order
        input_dict = {
            'Age': age,
            'Gender': gender,
            'Blood Type': blood_type,
            'Medical Condition': medical_condition,
            'Insurance Provider': insurance_provider,
            'Billing Amount': billing_amount,
            'Room Number': room_number,
            'Admission Type': admission_type,
            'Medication': medication,
            'Length_of_Stay': length_of_stay,
            'Admission_Year': admission_year,
            'Admission_Month': admission_month,
            'Admission_DayOfWeek': admission_dayofweek,
        }
        input_df = pd.DataFrame([input_dict])

        # Ensure correct column order
        feature_cols = meta['feature_columns']
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        try:
            pred_encoded = pipeline.predict(input_df)
            pred_label = le.inverse_transform(pred_encoded)[0]
            pred_proba = pipeline.predict_proba(input_df)[0]

            # Result display
            st.markdown("---")
            if pred_label == "Normal":
                st.success(f"## Predicted Test Result: {pred_label}")
            elif pred_label == "Abnormal":
                st.error(f"## Predicted Test Result: {pred_label}")
            else:
                st.warning(f"## Predicted Test Result: {pred_label}")

            # Probabilities
            st.subheader("Prediction Probabilities")
            classes = le.classes_
            prob_df = pd.DataFrame({
                'Test Result': classes,
                'Probability': [f"{p:.2%}" for p in pred_proba]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            # Bar chart of probabilities
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['#DD8452' if c == 'Abnormal' else '#4C72B0' if c == 'Normal' else '#8E7DC6' for c in classes]
            ax.barh(classes, pred_proba, color=colors, edgecolor='white')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Prediction Confidence')
            for i, p in enumerate(pred_proba):
                ax.text(p + 0.01, i, f'{p:.2%}', va='center')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Disclaimer
            st.info(
                "**Disclaimer:** This prediction is generated by a machine learning model "
                "trained on synthetic data and is for educational/demonstration purposes only. "
                "It should NOT be used for actual clinical decision-making."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    st.subheader(f"Best Model: {best_model_name}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{best_metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{best_metrics['precision']:.2%}")
    col3.metric("Recall",    f"{best_metrics['recall']:.2%}")
    col4.metric("F1-Score",  f"{best_metrics['f1_score']:.2%}")

    st.metric("ROC-AUC (macro OvR)", str(best_metrics['roc_auc']))

    st.subheader("All Model Comparison")
    st.dataframe(comparison_df.style.highlight_max(
        subset=['Accuracy', 'F1-Score'], color='#d4edda'
    ), use_container_width=True)

    # Comparison chart
    st.subheader("Model Comparison Chart")
    comp_fig_path = os.path.join(FIGURES, '12_model_comparison.png')
    if os.path.exists(comp_fig_path):
        st.image(comp_fig_path, use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion Matrix (Best Model)")
    cm_path = os.path.join(FIGURES, '13_confusion_matrix_best.png')
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)

    # Feature importance
    fi_path = os.path.join(FIGURES, '14_feature_importance.png')
    if os.path.exists(fi_path):
        st.subheader("Feature Importances")
        st.image(fi_path, use_container_width=True)

    st.subheader("Training Details")
    col1, col2 = st.columns(2)
    col1.metric("Training Samples", f"{best_metrics['train_size']:,}")
    col2.metric("Test Samples", f"{best_metrics['test_size']:,}")
    st.write(f"**Target Classes:** {', '.join(best_metrics['target_classes'])}")

# ════════════════════════════════════════════════════════════════════
# PAGE: VISUALISATIONS
# ════════════════════════════════════════════════════════════════════
elif page == "📉 Visualisations":
    st.title("📉 Data Visualisations")

    viz_list = [
        ("01_target_distribution.png",   "Target Class Distribution"),
        ("02_age_distribution.png",      "Age Distribution"),
        ("03_billing_distribution.png",  "Billing Amount Distribution"),
        ("04_gender_vs_test_results.png","Test Results by Gender"),
        ("05_condition_vs_test_results.png", "Test Results by Medical Condition"),
        ("06_admission_type.png",        "Admission Type Distribution"),
        ("07_length_of_stay.png",        "Length of Stay Distribution"),
        ("08_billing_by_test_results.png","Billing Amount by Test Result"),
        ("09_correlation_heatmap.png",   "Correlation Heatmap"),
        ("10_insurance_provider.png",    "Insurance Provider Distribution"),
        ("11_age_by_test_results.png",   "Age by Test Result"),
    ]

    for i in range(0, len(viz_list), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(viz_list):
                fname, title = viz_list[i + j]
                fpath = os.path.join(FIGURES, fname)
                if os.path.exists(fpath):
                    col.subheader(title)
                    col.image(fpath, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Healthcare ML Project | Built with Streamlit</small>",
    unsafe_allow_html=True
)
