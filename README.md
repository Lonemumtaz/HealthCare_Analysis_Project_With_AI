# Healthcare Test Results Prediction

## Overview

This project is a complete end-to-end Data Analytics and Machine Learning solution built on a real-world
**Healthcare Dataset** sourced from Kaggle. It predicts a patient's medical **Test Result** (Normal, Inconclusive,
or Abnormal) based on demographic and clinical features.

---

## Problem Statement

Given patient healthcare records including age, gender, medical condition, admission type, medication, billing
amount, and length of stay, predict whether the patient's test result will be **Normal**, **Inconclusive**, or
**Abnormal**.

---

## Objective

1. Understand and clean the healthcare dataset.
2. Perform exploratory data analysis with meaningful visualisations.
3. Engineer relevant features from raw data (e.g., length of stay from admission/discharge dates).
4. Train and compare multiple classification models.
5. Select the best model and save it.
6. Deploy an interactive Streamlit web application for predictions.

---

## Dataset

| Property | Value |
|---|---|
| Dataset Name | Healthcare Dataset |
| Source | Kaggle (Synthetic Healthcare Dataset) |
| Total Records | 55,500 |
| Features | 15 (including target) |
| Target Column | `Test Results` |
| Problem Type | **Multiclass Classification** |
| Target Classes | Normal, Inconclusive, Abnormal |

**Column Summary:**

| Column | Description |
|---|---|
| Name | Patient name |
| Age | Patient age |
| Gender | Male / Female |
| Blood Type | Patient blood type |
| Medical Condition | Primary diagnosis |
| Date of Admission | Admission date |
| Doctor | Attending physician |
| Hospital | Hospital name |
| Insurance Provider | Insurance company |
| Billing Amount | Total billing in USD |
| Room Number | Assigned room |
| Admission Type | Elective / Urgent / Emergency |
| Discharge Date | Discharge date |
| Medication | Prescribed drug |
| Test Results | **TARGET** — Normal / Inconclusive / Abnormal |

---

## Technologies Used

- **Python 3.10+**
- **pandas** — Data manipulation
- **numpy** — Numerical computation
- **matplotlib / seaborn** — Data visualisation
- **scikit-learn** — ML models, preprocessing, evaluation
- **joblib** — Model serialisation
- **Streamlit** — Interactive web frontend
- **Jupyter Notebook** — Interactive analysis
- **python-docx** — Word report generation

---

## Project Structure

```
project/
├── data/
│   └── dataset.csv                  # Healthcare dataset
├── notebooks/
│   └── Data_Analytics_ML_Project.ipynb  # Jupyter notebook
├── model/
│   ├── trained_model.pkl            # Saved best model pipeline
│   ├── label_encoder.pkl            # Label encoder
│   └── feature_meta.json            # Feature metadata for frontend
├── frontend/
│   └── app.py                       # Streamlit web application
├── outputs/
│   ├── figures/                     # All EDA and model charts
│   ├── metrics/                     # Model comparison CSV + JSON
│   └── predictions/                 # Test set predictions CSV
├── report/
│   └── Project_Report.docx          # Professional Word report
├── ml_pipeline.py                   # Full ML pipeline script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Methodology

1. **Data Inspection** — Shape, types, missing values, duplicates, distributions
2. **Data Cleaning** — Remove 534 duplicates, fix 108 negative billing amounts
3. **Feature Engineering** — Extract `Length_of_Stay`, `Admission_Year`, `Admission_Month`, `Admission_DayOfWeek` from date columns
4. **EDA** — 11 visualisations covering target distribution, demographics, medical conditions, correlations
5. **Preprocessing** — StandardScaler for numerics, OrdinalEncoder for categoricals, inside ColumnTransformer
6. **Model Training** — 5 models with stratified 80/20 split
7. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC
8. **Model Selection** — Best model chosen by F1-Score on test set
9. **Deployment** — Streamlit app with 5 pages

---

## Data Preprocessing

- Removed **534 duplicate rows** (final dataset: 54,966 rows)
- Fixed **108 negative Billing Amount** entries (took absolute value)
- Dropped high-cardinality columns: `Name`, `Doctor`, `Hospital`
- Extracted date features: `Length_of_Stay`, `Admission_Year`, `Admission_Month`, `Admission_DayOfWeek`
- Numerical features scaled with `StandardScaler`
- Categorical features encoded with `OrdinalEncoder`
- All preprocessing is inside a single `sklearn.Pipeline` to prevent data leakage

---

## EDA

Key findings:
- **Balanced classes**: Abnormal (18,627), Normal (18,517), Inconclusive (18,356)
- **Age**: Normally distributed, range 13–89, mean ~52
- **Billing Amount**: Uniformly distributed, range ~$0–$52,764
- **Length of Stay**: 1–30 days, mean ~15 days
- **Medical Conditions**: 6 conditions (Cancer, Obesity, Diabetes, Asthma, Hypertension, Arthritis)
- Low correlation between numerical features — confirms the synthetic nature

---

## Machine Learning Models

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| K-Nearest Neighbors | Non-parametric distance-based |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting |
| HistGradient Boosting | Fast histogram-based boosting |

Train/Test split: **80% / 20%** with stratification.

---

## Model Evaluation

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | **0.4338** | **0.4339** | **0.4338** | **0.4337** | **0.6336** |
| K-Nearest Neighbors | 0.3619 | 0.3626 | 0.3619 | 0.3601 | 0.5319 |
| HistGradient Boosting | 0.3571 | 0.3573 | 0.3571 | 0.3560 | 0.5268 |
| Gradient Boosting | 0.3358 | 0.3355 | 0.3358 | 0.3343 | 0.5042 |
| Logistic Regression | 0.3315 | 0.3318 | 0.3315 | 0.3278 | 0.5004 |

---

## Final Model

**Selected Model: Random Forest Classifier**

**Reason:**
- Highest accuracy (43.38%), highest F1-Score (43.37%), and highest ROC-AUC (0.6336)
- Substantially outperforms random baseline (33.3% for 3 balanced classes)
- Provides feature importances for interpretability
- Robust to mixed feature types

**Important Note:** This is a well-known synthetic Kaggle dataset where test result labels are
essentially randomly assigned. All models achieve near-chance performance, which is consistent
with published community analyses. The pipeline and methodology are correct and production-grade.

---

## How to Install

```bash
pip install -r requirements.txt
```

---

## How to Run the ML Pipeline

```bash
cd project
python ml_pipeline.py
```

---

## How to Run the Jupyter Notebook

```bash
cd project
jupyter notebook notebooks/Data_Analytics_ML_Project.ipynb
```

---

## How to Run the Frontend

```bash
cd project
streamlit run frontend/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Results

| Metric | Value |
|---|---|
| Best Model | Random Forest |
| Accuracy | 43.38% |
| Precision | 43.39% |
| Recall | 43.38% |
| F1-Score (weighted) | 43.37% |
| ROC-AUC (macro OvR) | 0.6336 |
| Training Samples | 43,972 |
| Test Samples | 10,994 |

---

## Future Scope

1. Try advanced ensemble methods (XGBoost, LightGBM, CatBoost)
2. Collect real (non-synthetic) healthcare data with genuine predictive signals
3. Add hyperparameter tuning (GridSearchCV / Optuna)
4. Implement SHAP explainability for individual predictions
5. Add SMOTE or other oversampling (currently classes are balanced)
6. Deploy as a REST API with FastAPI + Docker
7. Add authentication and logging to the Streamlit frontend

---

## Disclaimer

This project uses a **synthetic** healthcare dataset for educational/demonstration purposes only.
Predictions must NOT be used for real clinical decisions.
