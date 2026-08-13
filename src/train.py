import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall":    round(recall_score(y_test, preds), 4),
        "f1_score":  round(f1_score(y_test, preds), 4),
        "roc_auc":   round(roc_auc_score(y_test, probs), 4),
    }

def main():
    path = os.path.join(ROOT, "Data", "processed", "final_features.csv")
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "XGBoost":             XGBClassifier(
                                   n_estimators=200, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   use_label_encoder=False, eval_metric="logloss",
                                   random_state=42
                               ),
    }

    results = {}
    trained = {}
    print(f"\n{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 58)
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model
        print(f"{name:<22} {metrics['accuracy']:>6.4f} {metrics['precision']:>6.4f} "
              f"{metrics['recall']:>6.4f} {metrics['f1_score']:>6.4f} {metrics['roc_auc']:>6.4f}")

    # Select best by ROC-AUC
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = trained[best_name]
    print(f"\n✅ Best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.4f})")

    # Save artifacts
    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, "churn_model.pkl"))
    joblib.dump(best_name,  os.path.join(models_dir, "best_model_name.pkl"))

    # Save test split
    processed_dir = os.path.join(ROOT, "Data", "processed")
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    # Save comparison table
    results_dir = os.path.join(ROOT, "reports", "model_results")
    os.makedirs(results_dir, exist_ok=True)
    comp_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    comp_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    print(f"\nModel comparison saved.")

if __name__ == "__main__":
    main()
