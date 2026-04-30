import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def load_data():
    path = os.path.join(BASE_DIR, "data", "telco.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df):
    # Convert Total Charges to numeric
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # Fill missing Total Charges with median (IMPORTANT)
    df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())

    # Fill categorical missing values with "No"
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].fillna("No")

    return df


def drop_unnecessary_columns(df):
    drop_cols = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
        'Lat Long', 'Latitude', 'Longitude',
        'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'
    ]

    df = df.drop(columns=drop_cols)

    return df


def preprocess():
    df = load_data()
    df = clean_data(df)
    df = drop_unnecessary_columns(df)

    return df