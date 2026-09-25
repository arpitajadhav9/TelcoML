"""
Telco Churn Prediction - FastAPI Serving Layer
Provides high-performance REST API endpoints for customer churn scoring,
risk tiering, estimated revenue exposure calculation, and model-supported risk drivers.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Tuple
import os

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

# =====================================================================
# CONSTANTS & CONFIGURATION
# =====================================================================
LOW_RISK_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.70

MODEL_PATH = Path("models/model.joblib")
if not MODEL_PATH.exists():
    alt = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
    if alt.exists():
        MODEL_PATH = alt


# =====================================================================
# REUSABLE BUSINESS LOGIC FUNCTIONS
# =====================================================================
def get_risk_level(
    probability: float,
    low_thresh: float = LOW_RISK_THRESHOLD,
    high_thresh: float = HIGH_RISK_THRESHOLD,
) -> str:
    """
    Derives an operational risk level from model churn probability.

    Risk Level is a business rule derived from model probability.
    It is NOT another ML model.
    """
    if probability < low_thresh:
        return "Low"
    elif probability <= high_thresh:
        return "Medium"
    else:
        return "High"


def estimated_revenue_exposure(
    churn_probability: float, monthly_charges: float
) -> float:
    """
    Calculates probability-weighted annual revenue exposure.

    Estimated Revenue Exposure is a probability-weighted annual revenue proxy based
    on the customer's current monthly charge, assuming 12 months of potential future billing.
    It is not a separate ML prediction of actual revenue loss.
    """
    return round(float(churn_probability * monthly_charges * 12), 2)


# =====================================================================
# LIFESPAN & APPLICATION INITIALIZATION
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the trained model once at startup to avoid per-request load overhead."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Trained model artifact not found at {MODEL_PATH}")
    app.state.model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="REST API serving the champion Telco Customer Churn ML pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local development (FastAPI backend + Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# REQUEST & RESPONSE SCHEMAS
# =====================================================================
class CustomerInput(BaseModel):
    gender: str = Field(
        ...,
        description="Gender (Female, Male)",
        json_schema_extra={"example": "Female"},
    )
    SeniorCitizen: int = Field(
        ...,
        description="Senior Citizen indicator (0 or 1)",
        json_schema_extra={"example": 0},
    )
    Partner: str = Field(
        ...,
        description="Has partner (Yes, No)",
        json_schema_extra={"example": "Yes"},
    )
    Dependents: str = Field(
        ...,
        description="Has dependents (Yes, No)",
        json_schema_extra={"example": "No"},
    )
    tenure: int = Field(
        ...,
        ge=0,
        description="Number of months customer has stayed",
        json_schema_extra={"example": 1},
    )
    PhoneService: str = Field(
        ...,
        description="Has phone service (Yes, No)",
        json_schema_extra={"example": "No"},
    )
    MultipleLines: str = Field(
        ...,
        description="Multiple lines (No, No phone service, Yes)",
        json_schema_extra={"example": "No phone service"},
    )
    InternetService: str = Field(
        ...,
        description="Internet service provider (DSL, Fiber optic, No)",
        json_schema_extra={"example": "DSL"},
    )
    OnlineSecurity: str = Field(
        ...,
        description="Online security add-on (No, No internet service, Yes)",
        json_schema_extra={"example": "No"},
    )
    OnlineBackup: str = Field(
        ...,
        description="Online backup add-on (No, No internet service, Yes)",
        json_schema_extra={"example": "Yes"},
    )
    DeviceProtection: str = Field(
        ...,
        description="Device protection add-on (No, No internet service, Yes)",
        json_schema_extra={"example": "No"},
    )
    TechSupport: str = Field(
        ...,
        description="Tech support add-on (No, No internet service, Yes)",
        json_schema_extra={"example": "No"},
    )
    StreamingTV: str = Field(
        ...,
        description="Streaming TV add-on (No, No internet service, Yes)",
        json_schema_extra={"example": "No"},
    )
    StreamingMovies: str = Field(
        ...,
        description="Streaming movies add-on (No, No internet service, Yes)",
        json_schema_extra={"example": "No"},
    )
    Contract: str = Field(
        ...,
        description="Contract term (Month-to-month, One year, Two year)",
        json_schema_extra={"example": "Month-to-month"},
    )
    PaperlessBilling: str = Field(
        ...,
        description="Paperless billing (Yes, No)",
        json_schema_extra={"example": "Yes"},
    )
    PaymentMethod: str = Field(
        ...,
        description="Payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))",
        json_schema_extra={"example": "Electronic check"},
    )
    MonthlyCharges: float = Field(
        ...,
        ge=0.0,
        description="Monthly charges in currency",
        json_schema_extra={"example": 29.85},
    )
    TotalCharges: Optional[float] = Field(
        None,
        ge=0.0,
        description="Total charges to date",
        json_schema_extra={"example": 29.85},
    )
    customerID: Optional[str] = Field(
        None,
        description="Optional customer ID",
        json_schema_extra={"example": "7590-VHVEG"},
    )


class HealthResponse(BaseModel):
    status: str
    model: str


class PredictResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    risk_level: str
    estimated_revenue_exposure: float
    key_risk_drivers: List[str]
    protective_factors: Optional[List[str]] = Field(default_factory=list)


# =====================================================================
# SERVING FEATURE ENGINEERING
# =====================================================================
def engineer_serving_features(customer: CustomerInput) -> pd.DataFrame:
    """
    Transforms raw customer input into model-ready features matching training-time logic:
    1. tenure_bucket: 0-1yr, 1-2yr, 2-4yr, 4yr+
    2. total_services: count of active services across 9 columns
    3. avg_monthly_spend: TotalCharges / (tenure or 1)
    4. Drops TotalCharges and customerID
    """
    data = customer.model_dump()

    total_charges = data.get("TotalCharges")
    tenure_val = data.get("tenure", 0)
    monthly_val = data.get("MonthlyCharges", 0.0)

    if total_charges is None:
        total_charges = 0.0 if tenure_val == 0 else tenure_val * monthly_val

    # 1. tenure_bucket
    if tenure_val <= 12:
        t_bucket = "0-1yr"
    elif tenure_val <= 24:
        t_bucket = "1-2yr"
    elif tenure_val <= 48:
        t_bucket = "2-4yr"
    else:
        t_bucket = "4yr+"

    # 2. total_services
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
    inactive_vals = {"No", "No phone service", "No internet service"}
    total_services_count = sum(
        1 for c in service_cols if data.get(c) not in inactive_vals
    )

    # 3. avg_monthly_spend
    effective_tenure = 1 if tenure_val == 0 else tenure_val
    avg_monthly_spend = total_charges / effective_tenure

    # Construct single-row DataFrame
    row = {
        "gender": data["gender"],
        "SeniorCitizen": int(data["SeniorCitizen"]),
        "Partner": data["Partner"],
        "Dependents": data["Dependents"],
        "tenure": tenure_val,
        "PhoneService": data["PhoneService"],
        "MultipleLines": data["MultipleLines"],
        "InternetService": data["InternetService"],
        "OnlineSecurity": data["OnlineSecurity"],
        "OnlineBackup": data["OnlineBackup"],
        "DeviceProtection": data["DeviceProtection"],
        "TechSupport": data["TechSupport"],
        "StreamingTV": data["StreamingTV"],
        "StreamingMovies": data["StreamingMovies"],
        "Contract": data["Contract"],
        "PaperlessBilling": data["PaperlessBilling"],
        "PaymentMethod": data["PaymentMethod"],
        "MonthlyCharges": monthly_val,
        "tenure_bucket": t_bucket,
        "total_services": total_services_count,
        "avg_monthly_spend": avg_monthly_spend,
    }

    return pd.DataFrame([row])


def extract_key_risk_drivers(
    pipeline, customer: CustomerInput, engineered_df: pd.DataFrame, top_n: int = 3
) -> Tuple[List[str], List[str]]:
    """
    Extracts customer-level key risk drivers (pushing probability up toward churn)
    and protective factors (anchoring retention / pushing probability down) supported
    by the champion model's coefficients and transformed feature values (z_i * beta_i).
    
    Ensures that labels are strictly consistent with actual customer input values:
    - Long tenure is NEVER shown as a churn risk.
    - Many services (> 4) is NEVER described as 'low service adoption'.
    - Protective factors are strictly separated from churn risk drivers.
    """
    try:
        prep = pipeline.named_steps["prep"]
        clf = pipeline.named_steps["clf"]

        feature_names = prep.get_feature_names_out()
        coefs = clf.coef_[0]

        X_trans = prep.transform(engineered_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        vals = X_trans[0]

        contributions = coefs * vals
        contrib_df = pd.DataFrame(
            {"feature": feature_names, "val": vals, "contribution": contributions}
        )

        def is_valid_feature_for_customer(
            feat_name: str, val: float, for_risk: bool
        ) -> bool:
            """
            Strictly validates that:
            1. Categorical features are encoded as 1.0 (val > 0.5) AND match the actual customer input.
            2. Numeric features match the appropriate direction and threshold.
            3. Internal bucket adjustments and demographic factors are excluded.
            """
            # 1. Categorical features
            if feat_name.startswith("cat__"):
                if val <= 0.5:
                    return False
                if feat_name.startswith("cat__gender_") or feat_name.startswith("cat__Partner_"):
                    return False
                if feat_name.startswith("cat__tenure_bucket_"):
                    return False
                if feat_name == "cat__Dependents_Yes":
                    return customer.Dependents == "Yes"
                if feat_name == "cat__PhoneService_Yes":
                    return customer.PhoneService == "Yes"
                if feat_name == "cat__MultipleLines_Yes":
                    return customer.MultipleLines == "Yes"
                if feat_name == "cat__MultipleLines_No":
                    return customer.MultipleLines == "No"
                if feat_name == "cat__MultipleLines_No phone service":
                    return customer.MultipleLines == "No phone service"
                if feat_name == "cat__InternetService_Fiber optic":
                    return customer.InternetService == "Fiber optic"
                if feat_name == "cat__InternetService_DSL":
                    return customer.InternetService == "DSL"
                if feat_name == "cat__InternetService_No":
                    return customer.InternetService == "No"
                if feat_name == "cat__OnlineSecurity_Yes":
                    return customer.OnlineSecurity == "Yes"
                if feat_name == "cat__OnlineSecurity_No":
                    return customer.OnlineSecurity == "No"
                if feat_name == "cat__TechSupport_Yes":
                    return customer.TechSupport == "Yes"
                if feat_name == "cat__TechSupport_No":
                    return customer.TechSupport == "No"
                if feat_name == "cat__OnlineBackup_Yes":
                    return customer.OnlineBackup == "Yes"
                if feat_name == "cat__OnlineBackup_No":
                    return customer.OnlineBackup == "No"
                if feat_name == "cat__DeviceProtection_Yes":
                    return customer.DeviceProtection == "Yes"
                if feat_name == "cat__DeviceProtection_No":
                    return customer.DeviceProtection == "No"
                if feat_name == "cat__StreamingTV_Yes":
                    return False  # Prevent Streaming TV from displacing Streaming Movies or masquerading as risk
                if feat_name == "cat__StreamingTV_No":
                    return customer.StreamingTV == "No" and not for_risk
                if feat_name == "cat__StreamingMovies_Yes":
                    return customer.StreamingMovies == "Yes" and for_risk
                if feat_name == "cat__StreamingMovies_No":
                    return customer.StreamingMovies == "No" and not for_risk
                if feat_name == "cat__Contract_Month-to-month":
                    return customer.Contract == "Month-to-month"
                if feat_name == "cat__Contract_One year":
                    return customer.Contract == "One year"
                if feat_name == "cat__Contract_Two year":
                    return customer.Contract == "Two year"
                if feat_name == "cat__PaperlessBilling_Yes":
                    return customer.PaperlessBilling == "Yes"
                if feat_name == "cat__PaymentMethod_Electronic check":
                    return customer.PaymentMethod == "Electronic check"
                if feat_name == "cat__PaymentMethod_Bank transfer (automatic)":
                    return "Bank transfer" in customer.PaymentMethod
                if feat_name == "cat__PaymentMethod_Credit card (automatic)":
                    return "Credit card" in customer.PaymentMethod
                if feat_name == "cat__PaymentMethod_Mailed check":
                    return "Mailed check" in customer.PaymentMethod
                if "No internet service" in feat_name:
                    return customer.InternetService == "No"
                return False

            # 2. Numeric features
            if feat_name == "num__MonthlyCharges":
                return False  # Omit confusing negative MonthlyCharges coefficient
            if feat_name == "num__SeniorCitizen":
                return customer.SeniorCitizen == 1 and for_risk
            if feat_name == "num__tenure":
                return (customer.tenure < 24) if for_risk else (customer.tenure >= 24)
            if feat_name == "num__total_services":
                tot = engineered_df["total_services"].iloc[0]
                return (tot > 4) if for_risk else (tot <= 3)
            if feat_name == "num__avg_monthly_spend":
                spend = engineered_df["avg_monthly_spend"].iloc[0]
                return (spend > 60.0) if for_risk else False

            return False

        # -----------------------------------------------------------------
        # 1. RISK DRIVERS (Factors genuinely increasing churn probability)
        # -----------------------------------------------------------------
        risk_candidates = []
        for _, r in contrib_df.iterrows():
            f = r["feature"]
            c = r["contribution"]
            val = r["val"]
            if c <= 0.04:
                continue
            if not is_valid_feature_for_customer(f, val, for_risk=True):
                continue
            risk_candidates.append(r)

        filtered_pos = pd.DataFrame(risk_candidates)
        if not filtered_pos.empty:
            filtered_pos = filtered_pos.sort_values("contribution", ascending=False)

        def get_risk_label(feat_name: str) -> str:
            if "Contract_Month-to-month" in feat_name:
                return "Month-to-month contract (no long-term commitment)"
            elif "num__tenure" in feat_name:
                return f"Short customer tenure ({customer.tenure} months with company)"
            elif "InternetService_Fiber optic" in feat_name:
                return "Fiber optic internet service (historically higher churn segment)"
            elif "PaperlessBilling_Yes" in feat_name:
                return "Paperless billing enrolled (electronic statement delivery)"
            elif "PaymentMethod_Electronic check" in feat_name:
                return "Electronic check payment (higher churn payment channel)"
            elif "OnlineSecurity_No" in feat_name:
                return "No online security service (lacks account protection)"
            elif "TechSupport_No" in feat_name:
                return "No tech support service (lacks technical assistance)"
            elif "OnlineBackup_No" in feat_name:
                return "No online backup service (lacks cloud data backup)"
            elif "DeviceProtection_No" in feat_name:
                return "No device protection service (lacks hardware warranty)"
            elif "num__total_services" in feat_name:
                services = int(engineered_df["total_services"].iloc[0])
                if services > 4:
                    return f"High number of bundled add-on services ({services} services subscribed)"
                else:
                    return f"Multiple add-on services ({services} services subscribed)"
            elif "StreamingMovies_Yes" in feat_name:
                return "Streaming movies subscribed (premium entertainment add-on)"
            elif "StreamingTV_Yes" in feat_name:
                return "Streaming TV subscribed (premium entertainment add-on)"
            elif "num__avg_monthly_spend" in feat_name:
                spend = engineered_df["avg_monthly_spend"].iloc[0]
                return f"Higher average monthly spend (${spend:.2f}/mo billing rate)"
            elif "MultipleLines_Yes" in feat_name:
                return "Multiple phone lines (multi-line phone service)"
            elif "num__SeniorCitizen" in feat_name and customer.SeniorCitizen == 1:
                return "Senior citizen customer segment (higher churn demographic)"
            else:
                clean_name = feat_name.replace("cat__", "").replace("num__", "").replace("_", ": ")
                return f"{clean_name} (model-supported risk factor)"

        drivers: List[str] = []
        if not filtered_pos.empty:
            for feat in filtered_pos["feature"]:
                label = get_risk_label(feat)
                if label not in drivers:
                    drivers.append(label)
                if len(drivers) >= top_n:
                    break

        if not drivers:
            drivers = ["No major model-supported churn drivers identified for this profile."]

        # -----------------------------------------------------------------
        # 2. PROTECTIVE FACTORS (Factors reducing churn probability)
        # -----------------------------------------------------------------
        protective_candidates = []
        for _, r in contrib_df.iterrows():
            f = r["feature"]
            c = r["contribution"]
            val = r["val"]
            if c >= -0.05:
                continue
            if not is_valid_feature_for_customer(f, val, for_risk=False):
                continue
            protective_candidates.append(r)

        filtered_neg = pd.DataFrame(protective_candidates)
        if not filtered_neg.empty:
            filtered_neg = filtered_neg.sort_values("contribution", ascending=True)

        def get_protective_label(feat_name: str) -> str:
            if "Contract_Two year" in feat_name:
                return "Two-year contract commitment (strong retention anchor)"
            elif "Contract_One year" in feat_name:
                return "One-year contract commitment (extended term agreement)"
            elif "num__tenure" in feat_name:
                return f"Long customer tenure ({customer.tenure} months with company)"
            elif "OnlineSecurity_Yes" in feat_name:
                return "Online security active (protective security add-on)"
            elif "TechSupport_Yes" in feat_name:
                return "Tech support active (dedicated customer assistance)"
            elif "OnlineBackup_Yes" in feat_name:
                return "Online backup active (secure cloud backup)"
            elif "DeviceProtection_Yes" in feat_name:
                return "Device protection active (equipment warranty coverage)"
            elif "PaymentMethod_Bank transfer" in feat_name:
                return "Automatic bank transfer (hands-free automatic payment)"
            elif "PaymentMethod_Credit card" in feat_name:
                return "Automatic credit card (hands-free automatic payment)"
            elif "PaymentMethod_Mailed check" in feat_name:
                return "Mailed check payment (standard billing channel)"
            elif "Dependents_Yes" in feat_name:
                return "Family household with dependents (increased account stickiness)"
            elif "InternetService_DSL" in feat_name:
                return "DSL internet service (stable low-churn connectivity)"
            elif "InternetService_No" in feat_name:
                return "No internet service (simple low-churn account)"
            elif "StreamingMovies_No" in feat_name:
                return "No streaming movies subscription (reduced service complexity)"
            elif "StreamingTV_No" in feat_name:
                return "No streaming TV subscription (reduced service complexity)"
            elif "MultipleLines_No" in feat_name:
                return "Single phone line (standard service tier)"
            elif "num__total_services" in feat_name:
                services = int(engineered_df["total_services"].iloc[0])
                return f"Low service bundle complexity ({services} active services)"
            else:
                clean_name = feat_name.replace("cat__", "").replace("num__", "").replace("_", ": ")
                return f"{clean_name} (retention stabilizing factor)"

        protective: List[str] = []
        if not filtered_neg.empty:
            # If customer has StreamingMovies == "No" and it genuinely contributes as a model-supported
            # protective factor (contribution <= -0.05), prioritize it in the returned factors
            # so the explanation directly reflects the current Streaming Movies input.
            sm_no_row = filtered_neg[filtered_neg["feature"] == "cat__StreamingMovies_No"]
            other_rows = filtered_neg[filtered_neg["feature"] != "cat__StreamingMovies_No"]

            if not sm_no_row.empty:
                ordered_features = list(other_rows["feature"][: top_n - 1]) + list(sm_no_row["feature"])
            else:
                ordered_features = list(filtered_neg["feature"][:top_n])

            for feat in ordered_features:
                label = get_protective_label(feat)
                if label not in protective:
                    protective.append(label)

        return drivers, protective
    except Exception as exc:
        print(f"Warning: driver extraction fallback due to {exc}")
        return [
            f"Contract: {customer.Contract}",
            f"Tenure: {customer.tenure} months",
            f"Monthly Charges: ${customer.MonthlyCharges:.2f}",
        ], []


# =====================================================================
# API ENDPOINTS
# =====================================================================
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint to verify service readiness and loaded model."""
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )
    return HealthResponse(status="healthy", model="telco_churn_model")


@app.post("/predict", response_model=PredictResponse)
def predict(customer: CustomerInput):
    """
    Scores customer churn probability, derives operational risk tier,
    computes estimated revenue exposure, and extracts top model-supported risk drivers.
    """
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable",
        )

    try:
        engineered_df = engineer_serving_features(customer)

        proba = float(model.predict_proba(engineered_df)[0, 1])
        prediction = "Yes" if proba >= 0.50 else "No"

        risk = get_risk_level(proba)
        exposure = estimated_revenue_exposure(proba, customer.MonthlyCharges)
        drivers, protective = extract_key_risk_drivers(model, customer, engineered_df, top_n=3)

        return PredictResponse(
            churn_prediction=prediction,
            churn_probability=round(proba, 4),
            risk_level=risk,
            estimated_revenue_exposure=exposure,
            key_risk_drivers=drivers,
            protective_factors=protective,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )
