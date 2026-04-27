
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_url",      type=str)
parser.add_argument("--n_estimators",  type=int,   default=100)
parser.add_argument("--max_depth",     type=int,   default=5)
parser.add_argument("--min_samples_split", type=int, default=2)
parser.add_argument("--max_features", type=str,   default="sqrt")
args = parser.parse_args()

print("Data path received:", args.data_url)

# Load data
if os.path.isfile(args.data_url):
    df = pd.read_csv(args.data_url)
else:
    for f in os.listdir(args.data_url):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(args.data_url, f))
            break

print("✅ Data loaded, shape:", df.shape)

# Preprocessing
df = df.drop(columns=["customer_id"])
le = LabelEncoder()
df["country"] = le.fit_transform(df["country"])
df["gender"]  = le.fit_transform(df["gender"])

X = df.drop(columns=["churn"])
y = df["churn"]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run():
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("max_features",      args.max_features)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        max_features=args.max_features,
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
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ ROC AUC:   {roc_auc:.4f}")

    os.makedirs("outputs", exist_ok=True)
    pickle.dump(model, open("outputs/model.pkl", "wb"))
    mlflow.log_artifact("outputs/model.pkl")

print("✅ Training complete!")
