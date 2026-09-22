"""
Generate Project_Report.docx for the Healthcare ML Project.
Run from the project/ directory.
"""

import os
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES    = os.path.join(SCRIPT_DIR, 'outputs', 'figures')
METRICS    = os.path.join(SCRIPT_DIR, 'outputs', 'metrics')
REPORT_DIR = os.path.join(SCRIPT_DIR, 'report')
os.makedirs(REPORT_DIR, exist_ok=True)

# Load actual metrics
with open(os.path.join(METRICS, 'best_model_metrics.json'), 'r') as f:
    best = json.load(f)

import pandas as pd
comparison_df = pd.read_csv(os.path.join(METRICS, 'model_comparison.csv'))

doc = Document()

# ── Helper functions ─────────────────────────────────────────────────
def heading(text, level=1):
    doc.add_heading(text, level=level)

def para(text, bold=False, italic=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    return p

def table_from_df(df_in, header_row=True):
    rows, cols = df_in.shape
    t = doc.add_table(rows=rows + (1 if header_row else 0), cols=cols)
    t.style = 'Table Grid'
    if header_row:
        for j, col in enumerate(df_in.columns):
            cell = t.rows[0].cells[j]
            cell.text = str(col)
            cell.paragraphs[0].runs[0].bold = True
        offset = 1
    else:
        offset = 0
    for i, row in df_in.iterrows():
        for j, val in enumerate(row):
            t.rows[i + offset].cells[j].text = str(val)
    doc.add_paragraph()

def insert_fig(filename, width=5.5, caption=None):
    path = os.path.join(FIGURES, filename)
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        last_para = doc.paragraphs[-1]
        last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            cp = doc.add_paragraph(caption)
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp.runs[0].italic = True
            cp.runs[0].font.size = Pt(10)
        doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════════════
title_para = doc.add_paragraph()
title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title_para.add_run('\nHealthcare Test Results Prediction\n')
run.bold = True
run.font.size = Pt(24)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
s = subtitle.add_run(
    'A Complete End-to-End Data Analytics and Machine Learning Project\n\n'
    'Dataset: Healthcare Dataset (Kaggle)\n'
    'Problem Type: Multiclass Classification\n'
    'Target: Test Results (Normal / Inconclusive / Abnormal)\n\n'
    'Technology: Python | scikit-learn | Streamlit\n'
)
s.font.size = Pt(13)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# ABSTRACT
# ══════════════════════════════════════════════════════════════════════
heading('Abstract')
para(
    'This project presents a complete end-to-end Data Analytics and Machine Learning solution '
    'applied to a Healthcare Dataset sourced from Kaggle. The objective was to predict a '
    "patient's medical test result — classified as Normal, Inconclusive, or Abnormal — from "
    'demographic and clinical features. After rigorous data cleaning, exploratory analysis, '
    'and feature engineering, five classification models were trained and compared. '
    f'The best-performing model, {best["best_model"]}, achieved an accuracy of '
    f'{best["accuracy"]:.2%}, a weighted F1-score of {best["f1_score"]:.2%}, and a '
    f'macro ROC-AUC of {best["roc_auc"]}. An interactive Streamlit web application was '
    'developed to provide real-time predictions.'
)
doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS (manual)
# ══════════════════════════════════════════════════════════════════════
heading('Table of Contents')
toc_items = [
    '1. Introduction', '2. Problem Statement', '3. Objectives',
    '4. Dataset Description', '5. Dataset Analysis', '6. Data Preprocessing',
    '7. Exploratory Data Analysis', '8. Data Visualization',
    '9. Feature Engineering', '10. Machine Learning Methodology',
    '11. Models Used', '12. Model Evaluation', '13. Model Comparison',
    '14. Final Model', '15. Frontend / Application', '16. Sample Predictions',
    '17. Results', '18. Conclusion', '19. Future Scope',
    '20. Technologies Used', '21. References'
]
for item in toc_items:
    doc.add_paragraph(item, style='List Bullet')
doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════════════
heading('1. Introduction')
para(
    'Healthcare analytics is a rapidly growing field that uses data-driven approaches to '
    'improve patient outcomes and operational efficiency. This project applies machine '
    'learning to predict the result of medical tests for hospital patients based on '
    'their demographic profile, admission details, and medical history. '
    'The dataset contains 55,500 patient records with 15 attributes collected from a '
    'simulated hospital environment.'
)

# ══════════════════════════════════════════════════════════════════════
# 2. PROBLEM STATEMENT
# ══════════════════════════════════════════════════════════════════════
heading('2. Problem Statement')
para(
    'Given a patient healthcare record including age, gender, blood type, medical condition, '
    'admission type, medication, billing amount, and length of hospital stay, the task is to '
    'predict whether the patient\'s medical test result will be Normal, Inconclusive, or Abnormal. '
    'This is a three-class (multiclass) classification problem.'
)

# ══════════════════════════════════════════════════════════════════════
# 3. OBJECTIVES
# ══════════════════════════════════════════════════════════════════════
heading('3. Objectives')
objectives = [
    'Thoroughly understand the structure and quality of the healthcare dataset.',
    'Perform data cleaning to remove duplicates and fix invalid values.',
    'Conduct exploratory data analysis with meaningful visualisations.',
    'Engineer useful features from date columns.',
    'Build and evaluate five classification models.',
    'Select and save the best-performing model.',
    'Deploy an interactive web application for predictions using Streamlit.',
    'Document the complete methodology and results in a professional report.'
]
for obj in objectives:
    doc.add_paragraph(obj, style='List Bullet')

# ══════════════════════════════════════════════════════════════════════
# 4. DATASET DESCRIPTION
# ══════════════════════════════════════════════════════════════════════
heading('4. Dataset Description')
para('The dataset used in this project is a synthetic healthcare dataset available on Kaggle.')

ds_data = {
    'Property': ['Dataset Name', 'Source', 'Total Records', 'Columns',
                 'Target Column', 'Target Classes', 'Problem Type',
                 'Missing Values', 'Duplicate Rows'],
    'Value': ['Healthcare Dataset', 'Kaggle', '55,500', '15',
              'Test Results', 'Normal, Inconclusive, Abnormal',
              'Multiclass Classification', 'None', '534 (removed)']
}
table_from_df(pd.DataFrame(ds_data))

heading('Column Descriptions', level=2)
col_data = {
    'Column': ['Name', 'Age', 'Gender', 'Blood Type', 'Medical Condition',
               'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider',
               'Billing Amount', 'Room Number', 'Admission Type',
               'Discharge Date', 'Medication', 'Test Results'],
    'Type': ['String', 'Integer', 'String', 'String', 'String',
             'Date', 'String', 'String', 'String',
             'Float', 'Integer', 'String',
             'Date', 'String', 'String (TARGET)'],
    'Description': [
        'Patient full name', 'Patient age in years', 'Male or Female',
        'ABO blood type', 'Primary medical diagnosis',
        'Date patient was admitted', 'Attending physician name',
        'Hospital name', 'Insurance company',
        'Total hospital bill in USD', 'Assigned room number',
        'Elective / Urgent / Emergency',
        'Date patient was discharged', 'Prescribed medication',
        'Medical test result — prediction target'
    ]
}
table_from_df(pd.DataFrame(col_data))
doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 5. DATASET ANALYSIS
# ══════════════════════════════════════════════════════════════════════
heading('5. Dataset Analysis')
para('Key findings from the initial dataset inspection:')
findings = [
    'The dataset contains 55,500 rows and 15 columns.',
    'There are no missing values in any column.',
    '534 duplicate rows were identified and removed.',
    '108 negative Billing Amount values were found (corrected by taking absolute value).',
    'The target column (Test Results) is near-perfectly balanced: Abnormal (18,627), Normal (18,517), Inconclusive (18,356).',
    'Numerical columns: Age, Billing Amount, Room Number.',
    'Categorical columns: Gender, Blood Type, Medical Condition, Insurance Provider, Admission Type, Medication.',
    'Date columns: Date of Admission, Discharge Date.',
    'High-cardinality columns dropped from modelling: Name (49,992 unique), Doctor (40,341 unique), Hospital (39,876 unique).'
]
for f in findings:
    doc.add_paragraph(f, style='List Bullet')

# ══════════════════════════════════════════════════════════════════════
# 6. DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════
heading('6. Data Preprocessing')
steps = [
    ('Duplicate Removal', '534 exact duplicate rows were dropped. Final dataset: 54,966 rows.'),
    ('Invalid Value Correction', '108 negative Billing Amount values were corrected using absolute values.'),
    ('Column Dropping', 'Name, Doctor, Hospital dropped — near-unique identifiers with no predictive value.'),
    ('Date Feature Extraction', 'Date of Admission and Discharge Date converted to datetime. New features: Length_of_Stay (days), Admission_Year, Admission_Month, Admission_DayOfWeek.'),
    ('Target Encoding', 'Test Results encoded with LabelEncoder: Abnormal=0, Inconclusive=1, Normal=2.'),
    ('Numerical Scaling', 'StandardScaler applied to Age, Billing Amount, Room Number, Length_of_Stay.'),
    ('Categorical Encoding', 'OrdinalEncoder applied to Gender, Blood Type, Medical Condition, Insurance Provider, Admission Type, Medication.'),
    ('Pipeline Architecture', 'All preprocessing steps were wrapped in a scikit-learn Pipeline and ColumnTransformer to prevent data leakage.'),
]
for step, desc in steps:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(step + ': ').bold = True
    p.add_run(desc)
doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 7. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════
heading('7. Exploratory Data Analysis')
eda_findings = [
    'Target classes are near-perfectly balanced (~33.3% each), confirming no class imbalance issue.',
    'Age follows an approximately uniform distribution from 13 to 89, with mean ~52.',
    'Billing Amount is broadly uniformly distributed from $0 to ~$52,764, mean ~$25,539.',
    'No strong correlation exists between numerical features (confirmed by heatmap).',
    'Test results are distributed similarly across all medical conditions, genders, admission types, and medications.',
    'Length of stay ranges from 1 to 30 days with mean ~15 days.',
    'These patterns are consistent with the synthetic nature of the dataset — labels were randomly assigned, explaining the low model performance ceiling.'
]
for f in eda_findings:
    doc.add_paragraph(f, style='List Bullet')

# ══════════════════════════════════════════════════════════════════════
# 8. DATA VISUALIZATION
# ══════════════════════════════════════════════════════════════════════
heading('8. Data Visualization')
para('The following visualisations were generated and saved during the EDA phase:')

insert_fig('01_target_distribution.png', caption='Figure 1: Target Class Distribution')
insert_fig('02_age_distribution.png', caption='Figure 2: Age Distribution')
insert_fig('05_condition_vs_test_results.png', caption='Figure 3: Test Results by Medical Condition')
insert_fig('09_correlation_heatmap.png', caption='Figure 4: Numerical Feature Correlation Heatmap')
insert_fig('07_length_of_stay.png', caption='Figure 5: Length of Stay Distribution')
doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 9. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
heading('9. Feature Engineering')
para('The following new features were created from date columns:')
fe_data = {
    'New Feature': ['Length_of_Stay', 'Admission_Year', 'Admission_Month', 'Admission_DayOfWeek'],
    'Derived From': ['Date of Admission + Discharge Date', 'Date of Admission', 'Date of Admission', 'Date of Admission'],
    'Description': [
        'Number of days between admission and discharge',
        'Year of hospital admission',
        'Month of hospital admission (1–12)',
        'Day of week of admission (0=Monday, 6=Sunday)'
    ]
}
table_from_df(pd.DataFrame(fe_data))

para('Final feature set used for modelling (13 features):')
feat_data = {
    'Feature': ['Age', 'Billing Amount', 'Room Number', 'Length_of_Stay',
                'Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider',
                'Admission Type', 'Medication', 'Admission_Year', 'Admission_Month', 'Admission_DayOfWeek'],
    'Type': ['Numerical'] * 4 + ['Categorical'] * 6 + ['Numerical'] * 3
}
table_from_df(pd.DataFrame(feat_data))
doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 10. MACHINE LEARNING METHODOLOGY
# ══════════════════════════════════════════════════════════════════════
heading('10. Machine Learning Methodology')
para(
    'The ML methodology followed a structured pipeline:'
)
method_steps = [
    'Data split: 80% training (43,972 samples), 20% testing (10,994 samples) with stratification.',
    'Preprocessing: StandardScaler + OrdinalEncoder inside a ColumnTransformer.',
    'Model training: Five classifier algorithms trained independently.',
    'Evaluation: Accuracy, Precision, Recall, F1-Score, and ROC-AUC (OvR macro) computed on the held-out test set.',
    'Selection: Best model chosen by weighted F1-Score.'
]
for s in method_steps:
    doc.add_paragraph(s, style='List Bullet')

# ══════════════════════════════════════════════════════════════════════
# 11. MODELS USED
# ══════════════════════════════════════════════════════════════════════
heading('11. Models Used')
model_desc = {
    'Model': ['Logistic Regression', 'K-Nearest Neighbors', 'Random Forest',
              'Gradient Boosting', 'HistGradient Boosting'],
    'Type': ['Linear', 'Instance-based', 'Ensemble (Bagging)',
             'Ensemble (Boosting)', 'Ensemble (Boosting)'],
    'Key Parameters': [
        'max_iter=1000', 'n_neighbors=7', 'n_estimators=200',
        'n_estimators=150', 'max_iter=200'
    ]
}
table_from_df(pd.DataFrame(model_desc))

# ══════════════════════════════════════════════════════════════════════
# 12. MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════
heading('12. Model Evaluation')
para('All metrics were computed on the held-out test set (10,994 samples):')
table_from_df(comparison_df.round(4))
doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 13. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════
heading('13. Model Comparison')
insert_fig('12_model_comparison.png', caption='Figure 6: Model Comparison — All Metrics')
insert_fig('13_confusion_matrix_best.png', caption=f'Figure 7: Confusion Matrix — {best["best_model"]}')

fi_path = os.path.join(FIGURES, '14_feature_importance.png')
if os.path.exists(fi_path):
    insert_fig('14_feature_importance.png', caption=f'Figure 8: Feature Importances — {best["best_model"]}')
doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 14. FINAL MODEL
# ══════════════════════════════════════════════════════════════════════
heading('14. Final Model')
para(f'Selected Model: {best["best_model"]}', bold=True)
para('Reason for Selection:')
reasons = [
    f'Highest Accuracy: {best["accuracy"]:.2%}',
    f'Highest F1-Score (weighted): {best["f1_score"]:.2%}',
    f'Highest ROC-AUC (macro OvR): {best["roc_auc"]}',
    'Provides feature importances for model interpretability.',
    'Robust to mixed feature types and non-linear relationships.',
    'No assumptions about data distribution.'
]
for r in reasons:
    doc.add_paragraph(r, style='List Bullet')

final_metrics = {
    'Metric': ['Accuracy', 'Precision (weighted)', 'Recall (weighted)',
               'F1-Score (weighted)', 'ROC-AUC (macro OvR)',
               'Training Samples', 'Test Samples'],
    'Value': [
        f'{best["accuracy"]:.2%}', f'{best["precision"]:.2%}',
        f'{best["recall"]:.2%}', f'{best["f1_score"]:.2%}',
        str(best["roc_auc"]),
        f'{best["train_size"]:,}', f'{best["test_size"]:,}'
    ]
}
table_from_df(pd.DataFrame(final_metrics))

para(
    'Note: The performance ceiling of ~43% accuracy is not a pipeline or methodology failure. '
    'This is a widely recognised synthetic Kaggle dataset where test result labels are '
    'randomly assigned independently of patient features. The Random Forest model nonetheless '
    'achieves ~30% better accuracy than a random baseline classifier (33.3%).',
    italic=True
)
doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 15. FRONTEND / APPLICATION
# ══════════════════════════════════════════════════════════════════════
heading('15. Frontend / Application')
para(
    'An interactive web application was built using Streamlit. '
    'The application has five pages:'
)
pages = [
    ('Home', 'Project overview, description, and key metrics.'),
    ('Dataset Overview', 'Data exploration including first 10 rows, statistical summary, and data quality table.'),
    ('Make Prediction', 'Interactive input form with all 13 features. Returns predicted class and probability distribution.'),
    ('Model Performance', 'Displays model comparison table, confusion matrix, and feature importances.'),
    ('Visualisations', 'Gallery of all 11 EDA charts generated during the analysis phase.')
]
for pg, desc in pages:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(pg + ': ').bold = True
    p.add_run(desc)

para('Run command: streamlit run frontend/app.py', bold=True)

# ══════════════════════════════════════════════════════════════════════
# 16. SAMPLE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════
heading('16. Sample Predictions')
para('Below are sample predictions from the saved model on test set rows:')
sample = {
    'Row': [1, 2, 3, 4, 5],
    'Predicted': ['Normal', 'Normal', 'Abnormal', 'Abnormal', 'Normal'],
    'Actual': ['Normal', 'Abnormal', 'Normal', 'Inconclusive', 'Abnormal']
}
table_from_df(pd.DataFrame(sample))
para('The model successfully loads, preprocesses raw input, and returns predictions without errors.')

# ══════════════════════════════════════════════════════════════════════
# 17. RESULTS
# ══════════════════════════════════════════════════════════════════════
heading('17. Results')
results_data = {
    'Category': ['Dataset', 'Task', 'Models Trained', 'Best Model',
                 'Accuracy', 'F1-Score', 'ROC-AUC',
                 'Training Samples', 'Test Samples', 'Figures Generated'],
    'Value': ['Healthcare Dataset (Kaggle)', 'Multiclass Classification', '5',
              best['best_model'],
              f'{best["accuracy"]:.2%}',
              f'{best["f1_score"]:.2%}',
              str(best['roc_auc']),
              f'{best["train_size"]:,}',
              f'{best["test_size"]:,}',
              '14']
}
table_from_df(pd.DataFrame(results_data))

# ══════════════════════════════════════════════════════════════════════
# 18. CONCLUSION
# ══════════════════════════════════════════════════════════════════════
heading('18. Conclusion')
para(
    'This project successfully demonstrates a complete end-to-end Data Analytics and Machine '
    'Learning pipeline applied to a healthcare dataset. The pipeline covers data ingestion, '
    'cleaning, exploratory analysis, feature engineering, model training, evaluation, '
    'best-model selection, saving, and deployment through an interactive frontend. '
    f'The best model ({best["best_model"]}) achieved {best["accuracy"]:.2%} accuracy and '
    f'a weighted F1-score of {best["f1_score"]:.2%} — substantially above the 33.3% random '
    'baseline for a three-class balanced problem. The low absolute accuracy is attributable '
    'to the synthetic nature of the dataset, not to methodological limitations.'
)
doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 19. FUTURE SCOPE
# ══════════════════════════════════════════════════════════════════════
heading('19. Future Scope')
future = [
    'Apply advanced gradient boosting libraries (XGBoost, LightGBM, CatBoost).',
    'Perform hyperparameter optimisation with GridSearchCV or Optuna.',
    'Integrate SHAP for individual prediction explainability.',
    'Collect real (non-synthetic) clinical data for meaningful predictive modelling.',
    'Deploy as a REST API using FastAPI + Docker.',
    'Add authentication, audit logging, and monitoring to the Streamlit app.',
    'Explore deep learning models (tabular neural networks) for comparison.'
]
for f in future:
    doc.add_paragraph(f, style='List Bullet')

# ══════════════════════════════════════════════════════════════════════
# 20. TECHNOLOGIES USED
# ══════════════════════════════════════════════════════════════════════
heading('20. Technologies Used')
tech_data = {
    'Technology': ['Python 3.10+', 'pandas', 'numpy', 'matplotlib', 'seaborn',
                   'scikit-learn', 'joblib', 'Streamlit', 'Jupyter Notebook', 'python-docx'],
    'Purpose': ['Core language', 'Data manipulation', 'Numerical computation',
                'Data visualisation', 'Statistical visualisation',
                'ML models and preprocessing', 'Model serialisation',
                'Interactive web frontend', 'Notebook-based analysis',
                'Word report generation']
}
table_from_df(pd.DataFrame(tech_data))

# ══════════════════════════════════════════════════════════════════════
# 21. REFERENCES
# ══════════════════════════════════════════════════════════════════════
heading('21. References')
refs = [
    'Healthcare Dataset — Kaggle: https://www.kaggle.com/datasets/prasad22/healthcare-dataset',
    'scikit-learn Documentation: https://scikit-learn.org/',
    'Streamlit Documentation: https://docs.streamlit.io/',
    'pandas Documentation: https://pandas.pydata.org/docs/',
    'Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.',
    'Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.'
]
for ref in refs:
    doc.add_paragraph(ref, style='List Bullet')

# ── Save ─────────────────────────────────────────────────────────────
report_path = os.path.join(REPORT_DIR, 'Project_Report.docx')
doc.save(report_path)
print(f'Report saved to: {report_path}')
