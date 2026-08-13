import pandas as pd
import numpy as np
import joblib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_artifacts():
    models_dir = os.path.join(ROOT, "models")
    model        = joblib.load(os.path.join(models_dir, "churn_model.pkl"))
    scaler       = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    le_dict      = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
    feature_cols = joblib.load(os.path.join(models_dir, "feature_cols.pkl"))
    return model, scaler, le_dict, feature_cols

def preprocess_input(input_dict, le_dict, scaler, feature_cols):
    """
    Apply the same transformations used during training to a single input dict.
    input_dict: raw user inputs (pre-encoding, pre-scaling) with engineered features added.
    """
    df = pd.DataFrame([input_dict])

    # Encode categoricals using saved label encoders
    for col, le in le_dict.items():
        if col in df.columns:
            val = df[col].astype(str).values[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = le.transform([le.classes_[0]])  # fallback to first class

    # Align to training feature order (fill any missing with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_cols)
    return df_scaled

def predict_single(input_dict, model, scaler, le_dict, feature_cols):
    """
    input_dict: raw customer data including engineered fields (tenure_group, avg_monthly_spend).
    Returns (predicted_class, churn_probability).
    """
    df_ready = preprocess_input(input_dict, le_dict, scaler, feature_cols)
    pred = int(model.predict(df_ready)[0])
    prob = float(model.predict_proba(df_ready)[0][1])
    return pred, prob

def get_risk_category(prob):
    if prob < 0.35:
        return "Low", "green"
    elif prob < 0.60:
        return "Medium", "orange"
    else:
        return "High", "red"

if __name__ == "__main__":
    model, scaler, le_dict, feature_cols = load_artifacts()
    # Quick smoke test with first row of final_features
    df = pd.read_csv(os.path.join(ROOT, "Data", "processed", "final_features.csv"))
    sample = df.drop(columns=["Churn"]).iloc[0].to_dict()
    pred, prob = predict_single(sample, model, scaler, le_dict, feature_cols)
    print(f"Prediction: {'Churn' if pred else 'No Churn'}, Probability: {prob:.2%}")
