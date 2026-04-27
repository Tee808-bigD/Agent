
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_url",    type=str)
parser.add_argument("--output_dir",  type=str, default="outputs")
args = parser.parse_args()

# Load
if os.path.isfile(args.data_url):
    df = pd.read_csv(args.data_url)
else:
    for f in os.listdir(args.data_url):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(args.data_url, f))
            break

print("✅ Data loaded, shape:", df.shape)

# Preprocess
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

# Save outputs
os.makedirs(args.output_dir, exist_ok=True)
pd.DataFrame(X_train).to_csv(f"{args.output_dir}/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{args.output_dir}/X_test.csv",  index=False)
pd.DataFrame(y_train).to_csv(f"{args.output_dir}/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv(f"{args.output_dir}/y_test.csv",  index=False)

print("✅ Preprocessed data saved to:", args.output_dir)
