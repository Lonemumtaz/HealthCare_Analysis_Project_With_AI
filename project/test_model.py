import json, joblib, pandas as pd

MODEL_PATH = 'model/trained_model.pkl'
META_PATH  = 'model/feature_meta.json'
DATA_PATH  = 'data/dataset.csv'

artifact = joblib.load(MODEL_PATH)
pipe = artifact['pipeline']
le   = artifact['label_encoder']
best = artifact['best_model_name']
print(f'Model loaded: {best}')

with open(META_PATH) as f:
    meta = json.load(f)
print(f'Features: {meta["feature_columns"]}')

df = pd.read_csv(DATA_PATH)
print(f'Data loaded: {df.shape}')

# Build a sample input manually
sample = {
    'Age': 45,
    'Gender': 'Female',
    'Blood Type': 'O+',
    'Medical Condition': 'Diabetes',
    'Insurance Provider': 'Medicare',
    'Billing Amount': 22000.0,
    'Room Number': 305,
    'Admission Type': 'Urgent',
    'Medication': 'Aspirin',
    'Length_of_Stay': 12,
    'Admission_Year': 2022,
    'Admission_Month': 6,
    'Admission_DayOfWeek': 2,
}
X_sample = pd.DataFrame([sample])[meta['feature_columns']]
pred = le.inverse_transform(pipe.predict(X_sample))
proba = pipe.predict_proba(X_sample)[0]
print(f'Sample prediction: {pred[0]}')
print(f'Probabilities: {dict(zip(le.classes_, proba.round(3)))}')
print('All checks passed.')
