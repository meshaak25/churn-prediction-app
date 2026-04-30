import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from preprocess import preprocess
from features import create_features
from train import encode_data


def evaluate_model():
    # Load data
    df = preprocess()
    df = create_features(df)
    df = encode_data(df)

    # Split features & target
    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]

    # Load model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred))

    # 🔥 FEATURE IMPORTANCE
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feature_names = X.columns

        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        print("\nTop 10 Important Features:\n")
        print(feature_importance_df.head(10))

    else:
        print("\nModel does not support feature importance.")


if __name__ == "__main__":
    evaluate_model()