import joblib
import pandas as pd
import os
import numpy as np

# 1. Configuration
MODEL_DIR = './models'
MODEL_FILE = 'random_forest_model.joblib'
ENCODER_FILE = 'label_encoders.joblib'

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
ENCODER_PATH = os.path.join(MODEL_DIR, ENCODER_FILE)

FEATURE_NAMES = [
    'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
    'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 
    'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length'
]

# 2. Load Artifacts
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)
print("--- Model and encoders loaded ---")

# 3. Hardcode a new customer
customer_data = {
    'person_age': 25,
    'person_income': 25000,
    'person_home_ownership': 'RENT',
    'person_emp_length': 3.0,
    'loan_intent': 'PERSONAL',
    'loan_grade': 'B',
    'loan_amnt': 10000,
    'loan_int_rate': 10.99,
    'loan_percent_income': 0.15,
    'cb_person_default_on_file': 'N',
    'cb_person_cred_hist_length': 2
}

# 4. Create DataFrame and Preprocess
df = pd.DataFrame(customer_data, index=[0])

for col, le in encoders.items():
    text_value = df.loc[0, col]
    df[col] = le.transform([text_value])[0]

if pd.isna(df.loc[0, 'loan_int_rate']):
    df['loan_int_rate'] = 0.0
if pd.isna(df.loc[0, 'person_emp_length']):
    df['person_emp_length'] = 0.0

df_processed = df[FEATURE_NAMES]

# 5. Run Prediction (Directly)
prediction_value = model.predict(df_processed)[0] # Get the 0 or 1

# 6. Print Result
if prediction_value == 1:
    prediction_text = "Default (High Risk)"
else:
    prediction_text = "No Default (Low Risk)"

print("\n--- Prediction Result ---")
print(f"  Prediction: {prediction_text}")