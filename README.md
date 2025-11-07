# Credit Risk Predictor

This project implements a machine learning model to predict credit risk, helping to determine whether a loan applicant is likely to default or not. The model uses various personal and loan-related features to make predictions.

## Features

- Random Forest-based prediction model
- Support for various input features including:
  - Personal information (age, income, employment length)
  - Loan details (amount, interest rate, grade)
  - Credit history information
- Preprocessed data handling with label encoding
- Easy-to-use prediction interface

## Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ramacharya06/credit-risk-predictor.git
   cd credit-risk-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
credit-risk-predictor/
├── data/
│   ├── credit_risk_dataset.csv    # Original dataset
│   └── processed_data.csv         # Preprocessed dataset
├── models/
│   ├── label_encoders.joblib      # Saved label encoders
│   └── random_forest_model.joblib # Trained model
├── notebook/
│   ├── model_training.ipynb       # Model training notebook
│   └── preprocessing_dataset.ipynb # Data preprocessing notebook
├── main.py                        # Main prediction script
└── README.md
```

## Usage

1. The project comes with a pre-trained model. To make predictions for a new customer, modify the `customer_data` dictionary in `main.py` with the following information:

```python
customer_data = {
    'person_age': 25,                    # Age of the person
    'person_income': 25000,              # Annual income
    'person_home_ownership': 'RENT',     # RENT, OWN, MORTGAGE, or OTHER
    'person_emp_length': 3.0,            # Employment length in years
    'loan_intent': 'PERSONAL',           # Loan intention
    'loan_grade': 'B',                   # Loan grade (A to G)
    'loan_amnt': 10000,                 # Loan amount
    'loan_int_rate': 10.99,             # Interest rate
    'loan_percent_income': 0.15,        # Loan percent income
    'cb_person_default_on_file': 'N',   # Previous defaults (Y/N)
    'cb_person_cred_hist_length': 2     # Credit history length in years
}
```

2. Run the prediction:
   ```bash
   python main.py
   ```

## Model Training

If you want to retrain the model:

1. Open and run the notebooks in order:
   - First run `notebook/preprocessing_dataset.ipynb` to preprocess the data
   - Then run `notebook/model_training.ipynb` to train the model

## Dependencies

- pandas
- scikit-learn
- joblib
- numpy
- matplotlib
- seaborn
- ipykernel
- imbalanced-learn

