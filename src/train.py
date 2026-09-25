"""
Telco Customer Churn - Model Training Pipeline
Reproduces the final feature engineering, preprocessing, and champion model
from the exploratory notebook for execution as a DVC pipeline stage.
"""

import os
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

RANDOM_STATE = 42
DATASET_PATH = Path("Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"


def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the dataset and handles initial data integrity issues."""
    if not filepath.exists():
        # Fallback for execution from different working directories
        alt_path = Path(__file__).resolve().parent.parent / filepath
        if alt_path.exists():
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)

    # Clean TotalCharges: 11 customers with tenure=0 have blank TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the exact final feature engineering from the notebook."""
    df_fe = df.drop(columns=["customerID"]).copy()

    # 1. tenure_bucket (RFM-style tenure grouping)
    def tenure_bucket(t):
        if t <= 12:
            return "0-1yr"
        elif t <= 24:
            return "1-2yr"
        elif t <= 48:
            return "2-4yr"
        else:
            return "4yr+"

    df_fe["tenure_bucket"] = df_fe["tenure"].apply(tenure_bucket)

    # 2. total_services (Count of subscribed telecom services)
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def count_services(row):
        return sum(
            1
            for c in service_cols
            if row[c] not in ["No", "No phone service", "No internet service"]
        )

    df_fe["total_services"] = df_fe.apply(count_services, axis=1)

    # 3. avg_monthly_spend
    df_fe["avg_monthly_spend"] = df_fe["TotalCharges"] / df_fe["tenure"].replace(0, 1)

    # Drop collinear TotalCharges (TotalCharges = tenure * MonthlyCharges)
    df_fe = df_fe.drop(columns=["TotalCharges"])

    return df_fe


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Constructs the exact champion preprocessing and Logistic Regression pipeline."""
    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "avg_monthly_spend",
        "total_services",
        "SeniorCitizen",
    ]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )

    return pipeline


def train():
    """Executes the training workflow, evaluates test metrics, and exports champion model."""
    print("=" * 60)
    print("TELCO CHURN - MODEL TRAINING")
    print("=" * 60)

    # 1. Load data
    print(f"Loading dataset from: {DATASET_PATH}")
    df = load_data(DATASET_PATH)
    print(f"Raw dataset shape: {df.shape}")

    # 2. Feature engineering
    print("Applying feature engineering (tenure_bucket, total_services, avg_monthly_spend)...")
    df_fe = engineer_features(df)
    print(f"Engineered dataset shape: {df_fe.shape}")

    # 3. Target and feature split
    target = df_fe["Churn"].map({"Yes": 1, "No": 0})
    X = df_fe.drop(columns=["Churn"])
    y = target

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 5. Build and train champion pipeline
    print("Building champion pipeline (StandardScaler + OneHotEncoder + LogisticRegression)...")
    pipeline = build_pipeline(X_train)

    print("Fitting model...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
    }

    print("\n--- Model Evaluation Metrics (Test Set) ---")
    for metric_name, val in metrics.items():
        print(f"  {metric_name:<12}: {val:.4f}")

    # 7. Persist model artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nChampion model artifact successfully saved to: {MODEL_PATH}")
    print(f"Model step configuration: {list(pipeline.named_steps.keys())}")
    print(f"Classifier type: {type(pipeline.named_steps['clf']).__name__}")
    print("=" * 60)


if __name__ == "__main__":
    train()
