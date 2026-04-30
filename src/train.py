import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from preprocess import preprocess
from features import create_features


def encode_data(df):
    df = df.copy()
    encoders = {}

    for col in df.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def train_model():
    # Load + preprocess
    df = preprocess()
    df = create_features(df)

    # Encode
    df, encoders = encode_data(df)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Target distribution:\n", df["Churn Value"].value_counts())

    # Split
    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    lr = LogisticRegression(max_iter=2000)
    rf = RandomForestClassifier(random_state=42)

    lr.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)

    print("Logistic Regression Accuracy:", lr.score(X_test_scaled, y_test))
    print("Random Forest Accuracy:", rf.score(X_test, y_test))

    # Save model
    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(rf, f)

    # 🔥 Save encoders
    with open("models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    print("Model and encoders saved successfully!")


if __name__ == "__main__":
    train_model()