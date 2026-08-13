"""
Retention Simulator — estimates financial impact of targeting predicted churners
with a retention intervention campaign.
"""
import pandas as pd
import numpy as np
import joblib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def simulate_retention(
    X_scaled: pd.DataFrame,
    monthly_charges: pd.Series,
    model,
    intervention_cost_per_customer: float = 15.0,
    retention_success_rate: float = 0.30,
):
    """
    Parameters
    ----------
    X_scaled            : Pre-processed feature DataFrame (same format as training).
    monthly_charges     : Original MonthlyCharges values aligned to X_scaled rows.
    model               : Trained classifier.
    intervention_cost_per_customer : Cost (USD) to reach out to one at-risk customer.
    retention_success_rate         : Fraction of targeted customers we expect to retain.

    Returns a dict with simulation results.
    """
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

    n_total    = len(preds)
    n_at_risk  = int(preds.sum())
    n_retained = int(n_total - n_at_risk)

    # Revenue at risk (annualised)
    at_risk_charges = monthly_charges.values[preds == 1]
    annual_revenue_at_risk = float(at_risk_charges.sum() * 12)

    # Estimate revenue saved if we intervene
    revenue_saved = annual_revenue_at_risk * retention_success_rate

    # Cost of campaign
    total_intervention_cost = n_at_risk * intervention_cost_per_customer

    # ROI
    net_benefit = revenue_saved - total_intervention_cost
    roi = net_benefit / total_intervention_cost if total_intervention_cost > 0 else 0.0

    return {
        "total_customers":           n_total,
        "customers_at_risk":         n_at_risk,
        "customers_retained":        n_retained,
        "churn_rate_pct":            round(n_at_risk / n_total * 100, 1),
        "annual_revenue_at_risk":    round(annual_revenue_at_risk, 2),
        "revenue_saved_estimate":    round(revenue_saved, 2),
        "total_intervention_cost":   round(total_intervention_cost, 2),
        "net_benefit":               round(net_benefit, 2),
        "roi_pct":                   round(roi * 100, 1),
    }

if __name__ == "__main__":
    from src.feature_engineering import engineer_features, encode_and_scale

    model        = joblib.load(os.path.join(ROOT, "models", "churn_model.pkl"))
    le_dict      = joblib.load(os.path.join(ROOT, "models", "label_encoders.pkl"))
    scaler       = joblib.load(os.path.join(ROOT, "models", "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(ROOT, "models", "feature_cols.pkl"))

    df_clean = pd.read_csv(os.path.join(ROOT, "Data", "processed", "cleaned_data.csv"))
    monthly_charges = df_clean["MonthlyCharges"].reset_index(drop=True)

    df_feat = engineer_features(df_clean)
    X_scaled, y, _, _, _ = encode_and_scale(
        df_feat, fit=False, le_dict=le_dict, scaler=scaler, feature_cols=feature_cols
    )

    results = simulate_retention(X_scaled, monthly_charges, model)
    for k, v in results.items():
        print(f"  {k}: {v}")
