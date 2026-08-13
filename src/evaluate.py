import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    models_dir    = os.path.join(ROOT, "models")
    processed_dir = os.path.join(ROOT, "Data", "processed")
    figures_dir   = os.path.join(ROOT, "reports", "figures")
    results_dir   = os.path.join(ROOT, "reports", "model_results")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model      = joblib.load(os.path.join(models_dir, "churn_model.pkl"))
    model_name = joblib.load(os.path.join(models_dir, "best_model_name.pkl"))
    X_test     = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test     = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall":    round(recall_score(y_test, preds), 4),
        "f1_score":  round(f1_score(y_test, preds), 4),
        "roc_auc":   round(roc_auc_score(y_test, probs), 4),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print("Best model metrics:", metrics)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature":    X_test.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        fi.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roc_curve.png"), dpi=150)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    print("Evaluation complete. Figures saved.")

if __name__ == "__main__":
    main()
