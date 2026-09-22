"""
Healthcare Test Results Prediction ? Full ML Pipeline
======================================================
Target  : Test Results (Normal / Inconclusive / Abnormal)
Task    : Multiclass Classification
"""

import os, sys, warnings, json

# Ensure we always resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

FIGURES   = 'outputs/figures'
METRICS   = 'outputs/metrics'
PREDS_DIR = 'outputs/predictions'
MODEL_DIR = 'model'

for d in [FIGURES, METRICS, PREDS_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------
print("=" * 60)
print("STEP 1 ? LOAD DATASET")
print("=" * 60)

df = pd.read_csv('data/dataset.csv')
print(f"Shape          : {df.shape}")
print(f"Columns        : {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nStatistical summary:\n{df.describe()}")

# ---------------------------------------------
# 2. DATA QUALITY ANALYSIS
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 ? DATA QUALITY ANALYSIS")
print("=" * 60)

print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nNegative Billing Amount: {(df['Billing Amount'] < 0).sum()}")
print(f"\nTarget distribution:\n{df['Test Results'].value_counts()}")

# ---------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 ? DATA CLEANING")
print("=" * 60)

# Remove duplicates
df = df.drop_duplicates()
print(f"After dedup: {df.shape}")

# Fix billing negatives ? replace with absolute value
df['Billing Amount'] = df['Billing Amount'].abs()

# Standardise Name casing (not used in model, just clean for report)
df['Name'] = df['Name'].str.title()

# ---------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 ? FEATURE ENGINEERING")
print("=" * 60)

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date']    = pd.to_datetime(df['Discharge Date'])

df['Length_of_Stay']    = (df['Discharge Date'] - df['Date of Admission']).dt.days
df['Admission_Year']    = df['Date of Admission'].dt.year
df['Admission_Month']   = df['Date of Admission'].dt.month
df['Admission_DayOfWeek'] = df['Date of Admission'].dt.dayofweek

print(f"Length_of_Stay stats:\n{df['Length_of_Stay'].describe()}")
print(f"Sample rows:\n{df[['Length_of_Stay','Admission_Year','Admission_Month','Admission_DayOfWeek']].head()}")

# ---------------------------------------------
# 5. EDA  +  VISUALISATIONS
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 ? EDA & VISUALISATIONS")
print("=" * 60)

sns.set_style('whitegrid')

# 5a Target distribution
fig, ax = plt.subplots(figsize=(7, 4))
vc = df['Test Results'].value_counts()
ax.bar(vc.index, vc.values, color=['#4C72B0','#DD8452','#55A868'])
ax.set_title('Target Class Distribution ? Test Results', fontsize=14)
ax.set_xlabel('Test Result', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
for i, v in enumerate(vc.values):
    ax.text(i, v + 100, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{FIGURES}/01_target_distribution.png', dpi=120)
plt.close()
print("Saved: 01_target_distribution.png")

# 5b Age distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['Age'], bins=30, color='#4C72B0', edgecolor='white')
ax.set_title('Age Distribution', fontsize=14)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGURES}/02_age_distribution.png', dpi=120)
plt.close()
print("Saved: 02_age_distribution.png")

# 5c Billing Amount distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['Billing Amount'], bins=40, color='#DD8452', edgecolor='white')
ax.set_title('Billing Amount Distribution', fontsize=14)
ax.set_xlabel('Billing Amount ($)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGURES}/03_billing_distribution.png', dpi=120)
plt.close()
print("Saved: 03_billing_distribution.png")

# 5d Gender vs Test Results
fig, ax = plt.subplots(figsize=(7, 4))
cross = pd.crosstab(df['Gender'], df['Test Results'], normalize='index') * 100
cross.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white')
ax.set_title('Test Results by Gender (%)', fontsize=14)
ax.set_xlabel('Gender', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.legend(title='Test Result', bbox_to_anchor=(1, 1))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{FIGURES}/04_gender_vs_test_results.png', dpi=120)
plt.close()
print("Saved: 04_gender_vs_test_results.png")

# 5e Medical Condition vs Test Results
fig, ax = plt.subplots(figsize=(10, 5))
cross2 = pd.crosstab(df['Medical Condition'], df['Test Results'])
cross2.plot(kind='bar', ax=ax, colormap='Set1', edgecolor='white')
ax.set_title('Test Results by Medical Condition', fontsize=14)
ax.set_xlabel('Medical Condition', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.legend(title='Test Result', bbox_to_anchor=(1, 1))
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{FIGURES}/05_condition_vs_test_results.png', dpi=120)
plt.close()
print("Saved: 05_condition_vs_test_results.png")

# 5f Admission Type distribution
fig, ax = plt.subplots(figsize=(6, 4))
atype = df['Admission Type'].value_counts()
ax.bar(atype.index, atype.values, color=['#4C72B0','#DD8452','#55A868'], edgecolor='white')
ax.set_title('Admission Type Distribution', fontsize=14)
ax.set_xlabel('Admission Type', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGURES}/06_admission_type.png', dpi=120)
plt.close()
print("Saved: 06_admission_type.png")

# 5g Length of Stay distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['Length_of_Stay'], bins=30, color='#55A868', edgecolor='white')
ax.set_title('Length of Stay Distribution (Days)', fontsize=14)
ax.set_xlabel('Days', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGURES}/07_length_of_stay.png', dpi=120)
plt.close()
print("Saved: 07_length_of_stay.png")

# 5h Boxplot: Billing Amount by Test Results
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column='Billing Amount', by='Test Results', ax=ax, 
           boxprops=dict(color='#4C72B0'), medianprops=dict(color='red'))
ax.set_title('Billing Amount by Test Result', fontsize=14)
ax.set_xlabel('Test Result', fontsize=12)
ax.set_ylabel('Billing Amount ($)', fontsize=12)
plt.suptitle('')
plt.tight_layout()
plt.savefig(f'{FIGURES}/08_billing_by_test_results.png', dpi=120)
plt.close()
print("Saved: 08_billing_by_test_results.png")

# 5i Correlation heatmap (numeric only)
fig, ax = plt.subplots(figsize=(7, 5))
num_cols = ['Age', 'Billing Amount', 'Room Number', 'Length_of_Stay',
            'Admission_Year', 'Admission_Month', 'Admission_DayOfWeek']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Correlation Heatmap ? Numerical Features', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGURES}/09_correlation_heatmap.png', dpi=120)
plt.close()
print("Saved: 09_correlation_heatmap.png")

# 5j Insurance Provider distribution
fig, ax = plt.subplots(figsize=(8, 4))
ins = df['Insurance Provider'].value_counts()
ax.bar(ins.index, ins.values, color='#C44E52', edgecolor='white')
ax.set_title('Insurance Provider Distribution', fontsize=14)
ax.set_xlabel('Provider', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{FIGURES}/10_insurance_provider.png', dpi=120)
plt.close()
print("Saved: 10_insurance_provider.png")

# 5k Age boxplot by Test Results
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column='Age', by='Test Results', ax=ax,
           boxprops=dict(color='#4C72B0'), medianprops=dict(color='red'))
ax.set_title('Age by Test Result', fontsize=14)
ax.set_xlabel('Test Result', fontsize=12)
ax.set_ylabel('Age', fontsize=12)
plt.suptitle('')
plt.tight_layout()
plt.savefig(f'{FIGURES}/11_age_by_test_results.png', dpi=120)
plt.close()
print("Saved: 11_age_by_test_results.png")

# ---------------------------------------------
# 6. PREPARE FEATURES & ENCODE TARGET
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 6 ? FEATURE PREPARATION")
print("=" * 60)

# Drop columns not useful for modelling
drop_cols = ['Name', 'Doctor', 'Hospital', 'Date of Admission', 'Discharge Date']
df_model = df.drop(columns=drop_cols)

# Encode target
le = LabelEncoder()
df_model['Test Results Encoded'] = le.fit_transform(df_model['Test Results'])
TARGET = 'Test Results Encoded'
target_classes = le.classes_
print(f"Target classes (encoded): {dict(zip(le.classes_, le.transform(le.classes_)))}")

X = df_model.drop(columns=['Test Results', 'Test Results Encoded'])
y = df_model['Test Results Encoded']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape        : {y.shape}")
print(f"Features used       :\n{X.columns.tolist()}")

# Identify numerical and categorical columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'str']).columns.tolist()
print(f"\nNumerical features : {num_features}")
print(f"Categorical features: {cat_features}")

# ---------------------------------------------
# 7. TRAIN / TEST SPLIT
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 7 ? TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set : {X_train.shape}")
print(f"Test set     : {X_test.shape}")
print(f"Train class dist:\n{pd.Series(y_train).map(dict(enumerate(target_classes))).value_counts()}")
print(f"Test class dist:\n{pd.Series(y_test).map(dict(enumerate(target_classes))).value_counts()}")

# ---------------------------------------------
# 8. BUILD PREPROCESSING PIPELINE
# ---------------------------------------------
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# ---------------------------------------------
# 9. TRAIN MULTIPLE MODELS
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 8 ? TRAIN MULTIPLE MODELS")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=42),
    'HistGradient Boosting': HistGradientBoostingClassifier(max_iter=200, random_state=42),
}

results = {}

for name, estimator in models.items():
    print(f"\nTraining: {name} ...")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC-AUC (OvR, macro)
    try:
        y_prob = pipe.predict_proba(X_test)
        roc    = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    except Exception:
        roc = np.nan

    results[name] = {
        'pipeline' : pipe,
        'y_pred'   : y_pred,
        'Accuracy' : round(acc,  4),
        'Precision': round(prec, 4),
        'Recall'   : round(rec,  4),
        'F1-Score' : round(f1,   4),
        'ROC-AUC'  : round(roc,  4) if not np.isnan(roc) else 'N/A',
    }
    print(f"  Accuracy : {acc:.4f}  |  F1 : {f1:.4f}  |  ROC-AUC : {roc:.4f}")

# ---------------------------------------------
# 10. MODEL COMPARISON
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 9 ? MODEL COMPARISON")
print("=" * 60)

comparison_data = []
for name, r in results.items():
    comparison_data.append({
        'Model'    : name,
        'Accuracy' : r['Accuracy'],
        'Precision': r['Precision'],
        'Recall'   : r['Recall'],
        'F1-Score' : r['F1-Score'],
        'ROC-AUC'  : r['ROC-AUC'],
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('F1-Score', ascending=False)
comparison_df.to_csv(f'{METRICS}/model_comparison.csv', index=False)
print(comparison_df.to_string(index=False))

# Comparison bar chart
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(comparison_df))
width = 0.18
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    vals = comparison_df[metric].astype(float).values
    ax.bar(x + i*width, vals, width, label=metric, color=color, edgecolor='white')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comparison_df['Model'], rotation=12, ha='right', fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_title('Model Comparison ? Evaluation Metrics', fontsize=14)
ax.set_ylabel('Score', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{FIGURES}/12_model_comparison.png', dpi=120)
plt.close()
print("Saved: 12_model_comparison.png")

# ---------------------------------------------
# 11. SELECT BEST MODEL
# ---------------------------------------------
best_name = comparison_df.iloc[0]['Model']
best_pipe  = results[best_name]['pipeline']
best_preds = results[best_name]['y_pred']
print(f"\nBest model (by F1-Score): {best_name}")

# Detailed classification report
print("\nClassification Report (Best Model):")
print(classification_report(y_test, best_preds, target_names=target_classes))

# Confusion matrix
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=target_classes, yticklabels=target_classes)
ax.set_title(f'Confusion Matrix ? {best_name}', fontsize=13)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGURES}/13_confusion_matrix_best.png', dpi=120)
plt.close()
print("Saved: 13_confusion_matrix_best.png")

# Feature importance (if Random Forest / GBT)
if hasattr(best_pipe.named_steps['classifier'], 'feature_importances_'):
    fi = best_pipe.named_steps['classifier'].feature_importances_
    try:
        feat_names = (
            num_features +
            best_pipe.named_steps['preprocessor']
                .transformers_[1][1]
                .named_steps['encoder']
                .get_feature_names_out(cat_features).tolist()
        )
    except Exception:
        feat_names = [f'feature_{i}' for i in range(len(fi))]
    fi_df = pd.DataFrame({'Feature': feat_names[:len(fi)], 'Importance': fi})
    fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color='#4C72B0', edgecolor='white')
    ax.set_title(f'Top Feature Importances ? {best_name}', fontsize=13)
    ax.set_xlabel('Importance', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/14_feature_importance.png', dpi=120)
    plt.close()
    print("Saved: 14_feature_importance.png")

# ---------------------------------------------
# 12. SAVE BEST MODEL PIPELINE
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 10 ? SAVE MODEL")
print("=" * 60)

model_path = f'{MODEL_DIR}/trained_model.pkl'
joblib.dump({'pipeline': best_pipe, 'label_encoder': le,
             'feature_columns': X.columns.tolist(),
             'target_classes': target_classes.tolist(),
             'best_model_name': best_name}, model_path)
print(f"Model saved to: {model_path}")

# Save label encoder separately (for frontend use)
joblib.dump(le, f'{MODEL_DIR}/label_encoder.pkl')

# Save feature metadata for frontend
meta = {
    'feature_columns': X.columns.tolist(),
    'num_features': num_features,
    'cat_features': cat_features,
    'target_classes': target_classes.tolist(),
    'best_model_name': best_name,
    'cat_unique_values': {col: sorted(df[col].dropna().unique().tolist()) for col in cat_features},
}
with open(f'{MODEL_DIR}/feature_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f"Feature metadata saved to: {MODEL_DIR}/feature_meta.json")

# ---------------------------------------------
# 13. TEST SAVED MODEL
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 11 ? TEST SAVED MODEL")
print("=" * 60)

loaded = joblib.load(model_path)
loaded_pipe = loaded['pipeline']
loaded_le   = loaded['label_encoder']

# Use first 5 rows of test set as sample input
sample_input = X_test.iloc[:5].copy()
sample_pred_encoded = loaded_pipe.predict(sample_input)
sample_pred_labels  = loaded_le.inverse_transform(sample_pred_encoded)
sample_true_labels  = loaded_le.inverse_transform(y_test.iloc[:5].values)

print("Sample predictions from loaded model:")
for i, (pred, true) in enumerate(zip(sample_pred_labels, sample_true_labels)):
    print(f"  Row {i+1}: Predicted={pred:15s}  |  Actual={true}")

print("\n[OK] Saved model loaded and predictions verified successfully.")

# ---------------------------------------------
# 14. SAVE METRICS SUMMARY
# ---------------------------------------------
print("\n" + "=" * 60)
print("STEP 12 ? SAVE METRICS SUMMARY")
print("=" * 60)

best_metrics = {
    'best_model'   : best_name,
    'accuracy'     : float(results[best_name]['Accuracy']),
    'precision'    : float(results[best_name]['Precision']),
    'recall'       : float(results[best_name]['Recall']),
    'f1_score'     : float(results[best_name]['F1-Score']),
    'roc_auc'      : str(results[best_name]['ROC-AUC']),
    'train_size'   : int(X_train.shape[0]),
    'test_size'    : int(X_test.shape[0]),
    'total_records': int(df.shape[0]),
    'n_features'   : int(X.shape[1]),
    'target_classes': target_classes.tolist(),
}
with open(f'{METRICS}/best_model_metrics.json', 'w') as f:
    json.dump(best_metrics, f, indent=2)
print(f"Metrics saved to: {METRICS}/best_model_metrics.json")

# Also save predictions
pred_df = X_test.copy()
pred_df['True_Label']      = loaded_le.inverse_transform(y_test)
pred_df['Predicted_Label'] = loaded_le.inverse_transform(best_preds)
pred_df.to_csv(f'{PREDS_DIR}/test_predictions.csv', index=False)
print(f"Predictions saved to: {PREDS_DIR}/test_predictions.csv")

print("\n" + "=" * 60)
print("[DONE] ML PIPELINE COMPLETE")
print("=" * 60)
print(f"\nBest Model   : {best_name}")
print(f"Accuracy     : {results[best_name]['Accuracy']}")
print(f"F1-Score     : {results[best_name]['F1-Score']}")
print(f"ROC-AUC      : {results[best_name]['ROC-AUC']}")
print(f"\nAll figures  : {FIGURES}/")
print(f"All metrics  : {METRICS}/")
print(f"Saved model  : {model_path}")
