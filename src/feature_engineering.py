import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Categorical columns in the dataset (order matters for prediction alignment)
CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_group"
]

def engineer_features(df):
    df = df.copy()

    # Tenure buckets — useful signal, low-tenure customers churn more
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 60, 72],
            labels=["0-12m", "12-24m", "24-48m", "48-60m", "60-72m"]
        ).astype(str)

    # Average monthly spend — catches discrepancy between current and historical charges
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    return df

def encode_and_scale(df, target_col="Churn", fit=True, le_dict=None, scaler=None, feature_cols=None):
    df = df.copy()

    if fit:
        le_dict = {}
        for col in CAT_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
    else:
        for col, le in le_dict.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None

    if feature_cols is not None:
        X = X[feature_cols]

    if fit:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        feature_cols = list(X.columns)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    return X_scaled, y, le_dict, scaler, feature_cols

def main():
    in_path = os.path.join(ROOT, "Data", "processed", "cleaned_data.csv")
    df = pd.read_csv(in_path)
    df = engineer_features(df)

    X_scaled, y, le_dict, scaler, feature_cols = encode_and_scale(df, fit=True)

    final = X_scaled.copy()
    final["Churn"] = y.values
    out_path = os.path.join(ROOT, "Data", "processed", "final_features.csv")
    final.to_csv(out_path, index=False)

    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(le_dict, os.path.join(models_dir, "label_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_cols.pkl"))
    print(f"Final features saved: {final.shape}")
    print(f"Feature columns: {feature_cols}")

if __name__ == "__main__":
    main()
