import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Suppress sklearn pickle version warnings (no breaking changes for RF/LR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.predict import load_artifacts, predict_single, get_risk_category
from src.feature_engineering import engineer_features
from src.retention_simulator import simulate_retention

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2.2rem; font-weight:700; color:#1f3d6e; margin-bottom:0;}
    .subtitle   {font-size:1rem; color:#555; margin-top:0;}
    .kpi-card   {background:#f0f4fb; border-radius:10px; padding:18px 20px;
                 text-align:center; border-left:4px solid #4C72B0;}
    .kpi-value  {font-size:2rem; font-weight:700; color:#1f3d6e;}
    .kpi-label  {font-size:0.85rem; color:#666; margin-top:4px;}
    .result-box-churn    {background:#fff0f0; border:2px solid #e74c3c;
                          border-radius:10px; padding:20px; text-align:center;}
    .result-box-nochurn  {background:#f0fff4; border:2px solid #27ae60;
                          border-radius:10px; padding:20px; text-align:center;}
    .result-title {font-size:1.6rem; font-weight:700;}
    .insight-box {background:#fffbe6; border-left:4px solid #f0a500;
                  border-radius:6px; padding:12px 16px; margin-top:8px; font-size:0.9rem;}
    hr {border:none; border-top:1px solid #ddd; margin:16px 0;}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()

@st.cache_data(show_spinner=False)
def get_raw_data():
    return pd.read_csv(os.path.join(ROOT, "Data", "raw", "Telecom_churn.csv"))

@st.cache_data(show_spinner=False)
def get_clean_data():
    df = pd.read_csv(os.path.join(ROOT, "Data", "processed", "cleaned_data.csv"))
    # Normalise Churn to int — handles both "Yes"/"No" strings and 0/1 ints
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)
    return df

@st.cache_data(show_spinner=False)
def get_model_comparison():
    path = os.path.join(ROOT, "reports", "model_results", "model_comparison.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data(show_spinner=False)
def get_feature_importance():
    path = os.path.join(ROOT, "reports", "model_results", "feature_importance.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

model, scaler, le_dict, feature_cols = get_artifacts()
model_name = joblib.load(os.path.join(ROOT, "models", "best_model_name.pkl"))

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## 📡 Telecom Churn")
page = st.sidebar.radio(
    "Navigate to",
    ["🔮 Churn Prediction", "📊 Analytics Dashboard", "💰 Retention Simulator"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active model:** `{model_name}`")
st.sidebar.markdown("**Dataset:** Telco Customer Churn (IBM)")
st.sidebar.markdown("**Records:** 7,043 customers")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Churn Prediction":
    st.markdown('<p class="main-title">🔮 Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter customer details below and click <strong>Predict Churn</strong> to assess churn risk.</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**👤 Demographics**")
            gender          = st.selectbox("Gender",          ["Male", "Female"])
            senior_citizen  = st.selectbox("Senior Citizen",  ["No", "Yes"])
            partner         = st.selectbox("Has Partner",     ["Yes", "No"])
            dependents      = st.selectbox("Has Dependents",  ["No", "Yes"])

        with c2:
            st.markdown("**📞 Account & Charges**")
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
            total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                              round(monthly_charges * max(tenure, 1), 2), step=1.0)
            contract        = st.selectbox("Contract Type",
                                           ["Month-to-month", "One year", "Two year"])
            paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment         = st.selectbox("Payment Method", [
                                "Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"])

        with c3:
            st.markdown("**🌐 Services**")
            phone_service   = st.selectbox("Phone Service",    ["Yes", "No"])
            multi_lines     = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
            internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_sec      = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
            online_backup   = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
            device_prot     = st.selectbox("Device Protection",["No", "Yes", "No internet service"])
            tech_support    = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
            streaming_tv    = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
            streaming_mov   = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

    if submitted:
        # Build raw input dict then add engineered features
        raw = {
            "gender":           gender,
            "SeniorCitizen":    1 if senior_citizen == "Yes" else 0,
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           tenure,
            "PhoneService":     phone_service,
            "MultipleLines":    multi_lines,
            "InternetService":  internet,
            "OnlineSecurity":   online_sec,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_prot,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_mov,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
        }

        # Engineer features on single row
        tmp = pd.DataFrame([raw])
        tmp["tenure_group"]      = pd.cut(
            tmp["tenure"], bins=[-1, 12, 24, 48, 60, 72],
            labels=["0-12m", "12-24m", "24-48m", "48-60m", "60-72m"]
        ).astype(str)
        tmp["avg_monthly_spend"] = tmp["TotalCharges"] / tmp["tenure"].replace(0, 1)
        input_dict = tmp.iloc[0].to_dict()

        with st.spinner("Analyzing..."):
            pred, prob = predict_single(input_dict, model, scaler, le_dict, feature_cols)
            risk_cat, risk_color = get_risk_category(prob)

        st.markdown("<hr>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([2, 1, 1])

        with r1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-box-churn">
                    <div class="result-title">⚠️ Likely to Churn</div>
                    <p>This customer is at <strong style="color:{risk_color}">{risk_cat} Risk</strong> of leaving.</p>
                    <p style="font-size:0.9rem; color:#555;">
                    Consider proactive retention offers such as contract upgrades or discount incentives.
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-nochurn">
                    <div class="result-title">✅ Likely to Stay</div>
                    <p>This customer is at <strong style="color:{risk_color}">{risk_cat} Risk</strong> of churning.</p>
                    <p style="font-size:0.9rem; color:#555;">
                    No immediate action required. Monitor if charges or contract change.
                    </p>
                </div>""", unsafe_allow_html=True)

        with r2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#e74c3c" if prob > 0.6 else ("#f39c12" if prob > 0.35 else "#27ae60")},
                    "steps": [
                        {"range": [0, 35],  "color": "#d5f5e3"},
                        {"range": [35, 60], "color": "#fef9e7"},
                        {"range": [60, 100],"color": "#fadbd8"},
                    ]
                }
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with r3:
            st.markdown("**Risk Category**")
            st.markdown(f"<h2 style='color:{risk_color}'>{risk_cat}</h2>", unsafe_allow_html=True)
            st.markdown("**Churn Probability**")
            st.markdown(f"<h2>{prob:.1%}</h2>", unsafe_allow_html=True)
            st.markdown("**Prediction**")
            st.markdown(f"<h3>{'Churn' if pred else 'No Churn'}</h3>", unsafe_allow_html=True)

        # Key factors for this customer
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**📋 Customer Risk Factors**")
        flags = []
        if contract == "Month-to-month":
            flags.append("Month-to-month contract — highest churn association")
        if tenure <= 12:
            flags.append(f"Short tenure ({tenure} months) — new customers churn more")
        if monthly_charges > 70:
            flags.append(f"High monthly charges (${monthly_charges:.0f}) — above average")
        if payment == "Electronic check":
            flags.append("Electronic check payment — linked to higher churn rates")
        if internet == "Fiber optic" and online_sec == "No":
            flags.append("Fiber optic without Online Security — elevated risk")
        if not flags:
            flags.append("No major risk factors detected for this customer.")
        for f in flags:
            st.markdown(f'<div class="insight-box">⚠️ {f}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.markdown('<p class="main-title">📊 Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Insights from 7,043 telecom customers — understanding what drives churn.</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    df = get_clean_data().copy()
    df["Churn_label"] = df["Churn"].map({1: "Churn", 0: "No Churn"}).astype(str)

    # KPI cards
    total      = len(df)
    n_churn    = int(df["Churn"].sum())
    n_retain   = total - n_churn
    churn_rate = n_churn / total

    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in [
        (k1, f"{total:,}", "Total Customers"),
        (k2, f"{churn_rate:.1%}", "Overall Churn Rate"),
        (k3, f"{n_churn:,}", "Churned Customers"),
        (k4, f"{n_retain:,}", "Retained Customers"),
    ]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                     f'<div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Churn distribution + Contract ──────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        counts = df["Churn_label"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig = px.pie(counts, names="Status", values="Count",
                     color="Status",
                     color_discrete_map={"No Churn": "#4C72B0", "Churn": "#DD8452"},
                     hole=0.45)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 About <strong>26.5%</strong> of customers churned — significantly above typical industry churn of 15–20%, signalling a retention problem worth addressing.</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Churn by Contract Type")
        grp = (
                df.groupby(["Contract", "Churn_label"])
                .size()
                .reset_index(name="Count")
                )

        fig = px.bar(
            grp,
            x="Contract",
            y="Count",
            color="Churn_label",
            barmode="group",
            color_discrete_map={
            "No Churn": "#4C72B0",
            "Churn": "#DD8452"
            }
            )

        fig.update_layout(
            legend_title="",
            margin=dict(t=10, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

# Calculate month-to-month churn rate safely
        month_df = df[df["Contract"] == "Month-to-month"]

        if len(month_df) > 0:
            m2m_churn = (
                month_df["Churn_label"].eq("Churn").mean()
            )
        else:
            m2m_churn = 0

        st.markdown(
            f'<div class="insight-box">💡 <strong>{m2m_churn:.1%}</strong> '
            'of month-to-month customers churned vs single-digit rates for '
            'annual/biennial contracts. Contract length is the strongest '
            'retention lever.</div>',
            unsafe_allow_html=True
            )
    # ── Row 2: Payment method + Internet service ───────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn by Payment Method")
        grp = df.groupby(["PaymentMethod", "Churn_label"]).size().reset_index(name="Count")
        fig = px.bar(grp, x="Count", y="PaymentMethod", color="Churn_label",
                     orientation="h", barmode="group",
                     color_discrete_map={"No Churn": "#4C72B0", "Churn": "#DD8452"})
        fig.update_layout(legend_title="", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        ec_churn = df[df["PaymentMethod"]=="Electronic check"]["Churn"].mean()
        st.markdown(f'<div class="insight-box">💡 Electronic check users churn at <strong>{ec_churn:.1%}</strong> — the highest of all payment methods. Auto-payment options correlate with stronger retention.</div>', unsafe_allow_html=True)

    with col4:
        st.subheader("Churn by Internet Service")
        grp = df.groupby(["InternetService", "Churn_label"]).size().reset_index(name="Count")
        fig = px.bar(grp, x="InternetService", y="Count", color="Churn_label", barmode="group",
                     color_discrete_map={"No Churn": "#4C72B0", "Churn": "#DD8452"})
        fig.update_layout(legend_title="", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        fo_churn = df[df["InternetService"]=="Fiber optic"]["Churn"].mean()
        st.markdown(f'<div class="insight-box">💡 Fiber optic customers churn at <strong>{fo_churn:.1%}</strong> — possibly due to higher bills or competition. DSL customers are more stable.</div>', unsafe_allow_html=True)

    # ── Row 3: Tenure + Monthly charges ───────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Tenure Distribution by Churn")
        fig = px.histogram(df, x="tenure", color="Churn_label", nbins=30,
                           barmode="overlay", opacity=0.7,
                           color_discrete_map={"No Churn": "#4C72B0", "Churn": "#DD8452"})
        fig.update_layout(legend_title="", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        churn_med_tenure = df[df["Churn"]==1]["tenure"].median()
        retain_med_tenure = df[df["Churn"]==0]["tenure"].median()
        st.markdown(f'<div class="insight-box">💡 Churned customers have a median tenure of <strong>{churn_med_tenure:.0f} months</strong> vs <strong>{retain_med_tenure:.0f} months</strong> for retained customers — early months are the highest-risk window.</div>', unsafe_allow_html=True)

    with col6:
        st.subheader("Monthly Charges by Churn")
        fig = px.box(df, x="Churn_label", y="MonthlyCharges", color="Churn_label",
                     color_discrete_map={"No Churn": "#4C72B0", "Churn": "#DD8452"})
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        churn_med_mc = df[df["Churn"]==1]["MonthlyCharges"].median()
        retain_med_mc = df[df["Churn"]==0]["MonthlyCharges"].median()
        st.markdown(f'<div class="insight-box">💡 Churned customers pay a median of <strong>${churn_med_mc:.0f}/mo</strong> vs <strong>${retain_med_mc:.0f}/mo</strong> for retained customers — higher bills increase price-sensitivity.</div>', unsafe_allow_html=True)

    # ── Row 4: Feature importance + Demographics ───────────────────────────────
    col7, col8 = st.columns(2)

    with col7:
        st.subheader(f"Feature Importance — {model_name}")
        fi = get_feature_importance()
        if fi is not None:
            READABLE = {
                "Contract":          "Contract Type",
                "tenure":            "Tenure (months)",
                "TotalCharges":      "Total Charges",
                "MonthlyCharges":    "Monthly Charges",
                "avg_monthly_spend": "Avg Monthly Spend",
                "OnlineSecurity":    "Online Security",
                "tenure_group":      "Tenure Group",
                "TechSupport":       "Tech Support",
                "InternetService":   "Internet Service",
                "PaymentMethod":     "Payment Method",
                "gender":            "Gender",
                "SeniorCitizen":     "Senior Citizen",
                "Partner":           "Partner",
                "Dependents":        "Dependents",
                "PhoneService":      "Phone Service",
                "MultipleLines":     "Multiple Lines",
                "OnlineBackup":      "Online Backup",
                "DeviceProtection":  "Device Protection",
                "StreamingTV":       "Streaming TV",
                "StreamingMovies":   "Streaming Movies",
                "PaperlessBilling":  "Paperless Billing",
            }
            fi["label"] = fi["feature"].map(lambda x: READABLE.get(x, x))
            top10 = fi.head(10).sort_values("importance")
            fig = px.bar(top10, x="importance", y="label", orientation="h",
                         color="importance", color_continuous_scale="Blues")
            fig.update_layout(coloraxis_showscale=False,
                              yaxis_title="", xaxis_title="Importance Score",
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            top3 = fi.head(3)["label"].tolist()
            st.markdown(f'<div class="insight-box">💡 Top predictors: <strong>{", ".join(top3)}</strong>. Contract type and tenure dominate — confirming that relationship length and commitment level drive churn most.</div>', unsafe_allow_html=True)

    with col8:
        st.subheader("Churn by Senior Citizen Status")
        df["Senior"] = df["SeniorCitizen"].map({1: "Senior", 0: "Non-Senior"})
        grp = df.groupby(["Senior", "Churn_label"]).size().reset_index(name="Count")
        fig = px.bar(grp, x="Senior", y="Count", color="Churn_label", barmode="group",
                     color_discrete_map={"No Churn": "#4C72B0", "Churn": "#DD8452"})
        fig.update_layout(legend_title="", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        sr_churn = df[df["SeniorCitizen"]==1]["Churn"].mean()
        ns_churn = df[df["SeniorCitizen"]==0]["Churn"].mean()
        st.markdown(f'<div class="insight-box">💡 Senior citizens churn at <strong>{sr_churn:.1%}</strong> vs <strong>{ns_churn:.1%}</strong> for non-seniors — a meaningful demographic difference suggesting targeted support may help.</div>', unsafe_allow_html=True)

    # ── Model comparison table ─────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Model Comparison")
    comp = get_model_comparison()
    if comp is not None:
        comp_display = comp.copy()
        comp_display.columns = [c.replace("_", " ").title() for c in comp_display.columns]
        comp_display = comp_display.set_index("Model")
        # Highlight best model row
        best_idx = comp_display["Roc Auc"].idxmax()
        st.dataframe(
            comp_display.style
                .highlight_max(axis=0, color="#d5f5e3", subset=["Accuracy","Precision","Recall","F1 Score","Roc Auc"])
                .format("{:.4f}"),
            use_container_width=True
        )
        st.markdown(f'<div class="insight-box">✅ <strong>{best_idx}</strong> selected as production model based on highest ROC-AUC ({comp_display.loc[best_idx,"Roc Auc"]:.4f}). ROC-AUC is preferred over accuracy for imbalanced churn datasets.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RETENTION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Retention Simulator":
    st.markdown('<p class="main-title">💰 Retention Campaign Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Estimate the financial impact of targeting at-risk customers with a retention campaign.</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    **How this works:** The model identifies customers likely to churn. If the company invests in 
    a retention campaign (discounts, outreach, upgraded offers), a fraction of those customers 
    may be retained. This simulator estimates the revenue impact of such a campaign.
    """)

    col1, col2 = st.columns(2)
    with col1:
        intervention_cost = st.slider(
            "Cost per customer outreach ($)", 5, 100, 15,
            help="Cost to contact and offer a retention deal to one customer."
        )
    with col2:
        success_rate = st.slider(
            "Retention success rate (%)", 5, 60, 30,
            help="Estimated % of contacted at-risk customers who stay due to the campaign."
        ) / 100.0

    if st.button("▶ Run Simulation", use_container_width=True):
        with st.spinner("Running simulation on full dataset..."):
            from src.feature_engineering import encode_and_scale, engineer_features

            df_clean = get_clean_data()
            monthly_charges = df_clean["MonthlyCharges"].reset_index(drop=True)
            df_feat = engineer_features(df_clean)

            X_scaled, _, _, _, _ = encode_and_scale(
                df_feat, fit=False,
                le_dict=le_dict, scaler=scaler, feature_cols=feature_cols
            )

            results = simulate_retention(
                X_scaled, monthly_charges, model,
                intervention_cost_per_customer=intervention_cost,
                retention_success_rate=success_rate
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in [
            (m1, f"{results['customers_at_risk']:,}",          "Customers at Risk"),
            (m2, f"${results['annual_revenue_at_risk']:,.0f}", "Annual Revenue at Risk"),
            (m3, f"${results['revenue_saved_estimate']:,.0f}", "Estimated Revenue Saved"),
            (m4, f"{results['roi_pct']:.0f}%",                 "Campaign ROI"),
        ]:
            col.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                         f'<div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        net = results["net_benefit"]
        color = "#27ae60" if net > 0 else "#e74c3c"
        st.markdown(
            f'<div class="insight-box" style="border-color:{color}">💰 <strong>Net Benefit: ${net:,.0f}</strong> after subtracting ${results["total_intervention_cost"]:,.0f} campaign cost. '
            f'With a {int(success_rate*100)}% success rate and ${intervention_cost} cost per customer, '
            f'the campaign {"generates a positive return" if net > 0 else "does not break even at these parameters — try adjusting the sliders"}.</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure(go.Waterfall(
            name="Retention ROI",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Revenue at Risk", "Revenue Saved", "Campaign Cost", "Net Benefit"],
            y=[results["annual_revenue_at_risk"],
               results["revenue_saved_estimate"],
               -results["total_intervention_cost"],
               results["net_benefit"]],
            connector={"line": {"color": "#ccc"}},
            decreasing={"marker": {"color": "#DD8452"}},
            increasing={"marker": {"color": "#4C72B0"}},
            totals={"marker": {"color": "#27ae60" if net > 0 else "#e74c3c"}},
        ))
        fig.update_layout(title="Retention Campaign Waterfall", height=400,
                          yaxis_title="USD ($)", margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)