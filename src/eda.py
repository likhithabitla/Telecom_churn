"""
EDA — generates all report figures from the cleaned dataset.
Run once after preprocess.py. Figures saved to reports/figures/.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIGURES, exist_ok=True)

PALETTE = {"No Churn": "#4C72B0", "Churn": "#DD8452"}
sns.set_style("whitegrid")
sns.set_palette([PALETTE["No Churn"], PALETTE["Churn"]])

df = pd.read_csv(os.path.join(ROOT, "Data", "processed", "cleaned_data.csv"))
df["Churn_label"] = df["Churn"].map({1: "Churn", 0: "No Churn"})

# 1. Churn Distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Churn_label"].value_counts()
ax.bar(counts.index, counts.values, color=[PALETTE["No Churn"], PALETTE["Churn"]])
for i, (lbl, v) in enumerate(counts.items()):
    ax.text(i, v + 30, f"{v}\n({v/len(df):.1%})", ha="center", fontsize=10)
ax.set_title("Customer Churn Distribution", fontweight="bold")
ax.set_ylabel("Number of Customers")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "churn_distribution.png"), dpi=150); plt.close()

# 2. Contract vs Churn
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn_label", palette=PALETTE, ax=ax)
ax.set_title("Contract Type vs Churn", fontweight="bold")
ax.set_xlabel("Contract Type"); ax.set_ylabel("Count")
ax.legend(title=""); plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "contract_vs_churn.png"), dpi=150); plt.close()

# 3. Tenure vs Churn
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Churn_label", y="tenure", palette=PALETTE, ax=ax)
ax.set_title("Tenure (months) vs Churn", fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("Tenure (months)")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "tenure_vs_churn.png"), dpi=150); plt.close()

# 4. Monthly Charges vs Churn
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Churn_label", y="MonthlyCharges", palette=PALETTE, ax=ax)
ax.set_title("Monthly Charges vs Churn", fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("Monthly Charges ($)")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "monthly_charges_vs_churn.png"), dpi=150); plt.close()

# 5. Payment Method vs Churn
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=df, x="PaymentMethod", hue="Churn_label", palette=PALETTE, ax=ax)
ax.set_title("Payment Method vs Churn", fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.legend(title=""); plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "payment_method_vs_churn.png"), dpi=150); plt.close()

# 6. Internet Service vs Churn
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="InternetService", hue="Churn_label", palette=PALETTE, ax=ax)
ax.set_title("Internet Service vs Churn", fontweight="bold")
ax.legend(title=""); plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "internet_service_vs_churn.png"), dpi=150); plt.close()

# 7. Tech Support vs Churn
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="TechSupport", hue="Churn_label", palette=PALETTE, ax=ax)
ax.set_title("Tech Support vs Churn", fontweight="bold")
ax.legend(title=""); plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "techsupport_vs_churn.png"), dpi=150); plt.close()

# 8. Senior Citizen vs Churn
fig, ax = plt.subplots(figsize=(6, 4))
df["SeniorCitizen_label"] = df["SeniorCitizen"].map({1: "Senior", 0: "Non-Senior"})
sns.countplot(data=df, x="SeniorCitizen_label", hue="Churn_label", palette=PALETTE, ax=ax)
ax.set_title("Senior Citizen vs Churn", fontweight="bold")
ax.set_xlabel(""); ax.legend(title=""); plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "senior_vs_churn.png"), dpi=150); plt.close()

# 9. Correlation heatmap (numeric only)
num_df = pd.get_dummies(df.drop(columns=["Churn_label","SeniorCitizen_label"]))
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(num_df.corr(), cmap="coolwarm", ax=ax, linewidths=0.3)
ax.set_title("Correlation Heatmap", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "correlation_heatmap.png"), dpi=150); plt.close()

print("EDA complete. Figures saved to reports/figures/")
