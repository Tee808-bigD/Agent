
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",    type=str)
parser.add_argument("--output_dir",   type=str, default="outputs")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth",    type=int, default=5)
args = parser.parse_args()

# Load preprocessed data
X_train = pd.read_csv(f"{args.input_dir}/X_train.csv").values
X_test  = pd.read_csv(f"{args.input_dir}/X_test.csv").values
y_train = pd.read_csv(f"{args.input_dir}/y_train.csv").values.ravel()
y_test  = pd.read_csv(f"{args.input_dir}/y_test.csv").values.ravel()

print("✅ Data loaded from pipeline")

mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run():
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth",    args.max_depth)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("roc_auc",   roc_auc)

    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ ROC AUC:   {roc_auc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    pickle.dump(model, open(f"{args.output_dir}/model.pkl", "wb"))
    mlflow.log_artifact(f"{args.output_dir}/model.pkl")

print("✅ Pipeline training complete!")
