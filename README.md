# Telecom Customer Churn Prediction & Retention System

An end-to-end machine learning application that predicts customer churn for a telecom company, enabling early identification of customers likely to discontinue services and quantifying the financial impact of retention campaigns.

---

## Problem Statement

Telecom companies lose significant revenue to churn. Identifying at-risk customers before they leave allows targeted retention interventions. This project builds a full ML pipeline — from raw data to an interactive Streamlit dashboard — to solve this problem.

---

## Dataset

- **Source:** IBM Telco Customer Churn dataset
- **Records:** 7,043 customers
- **Target:** `Churn` (Yes / No → 1 / 0)
- **Churn Rate:** ~26.5%
- **Features:** 19 customer attributes (demographics, services, account info, charges)

---

## Project Structure

```
TelecomChurn/
├── Data/
│   ├── raw/Telecom_churn.csv
│   └── processed/
│       ├── cleaned_data.csv
│       ├── final_features.csv
│       ├── X_test.csv
│       └── y_test.csv
├── src/
│   ├── preprocess.py           # Data cleaning & type fixing
│   ├── feature_engineering.py  # Feature creation, encoding, scaling
│   ├── eda.py                  # EDA figures
│   ├── train.py                # Train & compare 4 models
│   ├── evaluate.py             # Metrics, ROC curve, confusion matrix
│   ├── predict.py              # Single-customer prediction utility
│   └── retention_simulator.py  # ROI simulation for campaigns
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_cols.pkl
│   └── best_model_name.pkl
├── app/
│   └── streamlit_app.py        # Three-page Streamlit app
├── reports/
│   ├── figures/                # All EDA & evaluation plots
│   └── model_results/
│       ├── model_comparison.csv
│       ├── metrics.csv
│       └── feature_importance.csv
├── requirements.txt
└── README.md
```

---

## ML Workflow

### 1. Preprocessing (`src/preprocess.py`)
- Convert `TotalCharges` from string to numeric (handles blank values)
- Drop duplicates and `customerID`
- Encode target: `Churn` → 0/1

### 2. Feature Engineering (`src/feature_engineering.py`)
- **tenure_group**: Bucketed tenure into 5 lifecycle stages (0-12m, 12-24m, 24-48m, 48-60m, 60-72m)
- **avg_monthly_spend**: `TotalCharges / tenure` — catches pricing inconsistencies
- Label encoding for all categorical features (fitted once on training data, reused for inference)
- StandardScaler applied to all features (same scaler reused for inference)

### 3. EDA (`src/eda.py`)
Generates figures for:
- Churn distribution, contract type, payment method, internet service
- Tenure and monthly charges distributions by churn
- Tech support, senior citizen demographics, correlation heatmap

### 4. Model Training (`src/train.py`)
All four models trained on 80/20 stratified split (random_state=42):

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 0.7999   | 0.6447    | 0.5481 | 0.5925 | 0.8405  |
| Decision Tree       | 0.7892   | 0.6170    | 0.5428 | 0.5775 | 0.8231  |
| **Random Forest**   | **0.7949** | **0.6523** | 0.4866 | 0.5574 | **0.8417** |
| XGBoost             | 0.7956   | 0.6453    | 0.5107 | 0.5701 | 0.8408  |

**Selected model: Random Forest** — highest ROC-AUC (0.8417), which is the most appropriate metric for this imbalanced classification task (26.5% churn rate).

> Note: XGBoost was very close (0.8408). The selection is data-driven and honest — if results differ on a different environment this is expected due to minor version differences.

### 5. Evaluation (`src/evaluate.py`)
- Full metrics report saved to `reports/model_results/metrics.csv`
- Feature importance saved to `reports/model_results/feature_importance.csv`
- ROC curve and confusion matrix saved to `reports/figures/`

---

## Key Business Insights (from actual data)

- **26.5%** overall churn rate
- **Month-to-month** contracts churn at ~43% vs ~11% for annual contracts
- **Electronic check** payment method has the highest churn rate (~45%)
- **Fiber optic** customers churn at ~42% — possibly due to higher bills
- Churned customers have **median tenure of 10 months** vs 38 months for retained
- Churned customers pay **~$79/mo** vs ~$61/mo for retained customers
- **Senior citizens** churn at ~41% vs ~24% for non-seniors

---

## Streamlit Application

Three pages:

### 🔮 Churn Prediction
- Full 19-field customer input form
- Churn probability gauge chart
- Risk category: Low / Medium / High
- Personalized risk factor explanations

### 📊 Analytics Dashboard
- KPI cards (total customers, churn rate, churned, retained)
- 8 interactive Plotly charts with business insight annotations
- Model comparison table
- Feature importance horizontal bar chart

### 💰 Retention Simulator
- Adjustable intervention cost and success rate sliders
- ROI calculation from model predictions
- Waterfall chart showing revenue at risk vs. savings vs. cost

---

## Installation & Running

```bash
# Clone and enter project directory
git clone <your-repo-url>
cd TelecomChurn

# Install dependencies
pip install -r requirements.txt

# Run the ML pipeline (only needed once, or when retraining)
python src/preprocess.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py

# Generate EDA figures (optional)
python src/eda.py

# Launch the app (from project root)
streamlit run app/streamlit_app.py
```

> **Important:** Always run from the project root directory, not from inside `src/` or `app/`.

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Plotly, Matplotlib, Seaborn, Joblib

---

## Resume Description

> Developed a machine learning application to predict customer churn using the Telecom Customer Churn dataset (7,043 records), enabling early identification of customers likely to discontinue services.
> Performed data preprocessing, EDA, feature engineering, and compared four classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost), selecting the best by ROC-AUC.
> Built a three-page interactive Streamlit application with a full customer prediction interface, analytics dashboard with business insights, and a retention ROI simulator.
> Achieved ROC-AUC of 0.8417 (Random Forest) and identified contract type and tenure as the strongest churn predictors.
