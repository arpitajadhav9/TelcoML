"""
Tests for FastAPI Serving Layer
Verifies health check, input validation, churn prediction schema, risk tiers, and revenue exposure.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import (
    app,
    get_risk_level,
    estimated_revenue_exposure,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
)


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient with lifespan initialization."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test 1: GET /health returns 200 with healthy status and model name."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "telco_churn_model"


def test_predict_endpoint_valid_customer(client):
    """Test 2 & 3: POST /predict with valid customer and verifies all required output fields."""
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    result = response.json()

    # Check all required fields are present
    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "risk_level" in result
    assert "estimated_revenue_exposure" in result
    assert "key_risk_drivers" in result

    # Check data types and value ranges
    assert result["churn_prediction"] in ["Yes", "No"]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert result["estimated_revenue_exposure"] >= 0.0
    assert isinstance(result["key_risk_drivers"], list)
    assert len(result["key_risk_drivers"]) >= 1


def test_predict_endpoint_low_risk_customer(client):
    """Test prediction for a low-risk loyal customer."""
    loyal_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2925.0,
    }

    response = client.post("/predict", json=loyal_customer)
    assert response.status_code == 200
    result = response.json()
    assert result["churn_prediction"] == "No"
    assert result["risk_level"] == "Low"
    assert result["churn_probability"] < 0.40


def test_predict_validation_error(client):
    """Test POST /predict returns 422 for missing required fields."""
    invalid_customer = {
        "gender": "Female",
        "tenure": 5,
        # Missing remaining required fields
    }
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422


def test_risk_level_function():
    """Verify rule-based risk level derivation."""
    assert get_risk_level(0.25) == "Low"
    assert get_risk_level(0.40) == "Medium"
    assert get_risk_level(0.55) == "Medium"
    assert get_risk_level(0.70) == "Medium"
    assert get_risk_level(0.75) == "High"


def test_estimated_revenue_exposure_function():
    """Verify formula: churn_probability * monthly_charges * 12."""
    prob = 0.50
    monthly = 100.0
    expected = 0.50 * 100.0 * 12  # 600.0
    assert estimated_revenue_exposure(prob, monthly) == 600.0
