import pandas as pd
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(path=None):
    if path is None:
        path = os.path.join(ROOT, "Data", "raw", "Telecom_churn.csv")
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Fix TotalCharges (often stored as string with spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop customer ID
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Encode target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df

def main():
    df = load_data()
    df_clean = clean_data(df)
    out_path = os.path.join(ROOT, "Data", "processed", "cleaned_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print(f"Cleaned data saved: {df_clean.shape}")
    print(f"Churn rate: {df_clean['Churn'].mean():.2%}")

if __name__ == "__main__":
    main()
