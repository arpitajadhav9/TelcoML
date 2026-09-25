"""
Telco Customer Churn Intelligence - Streamlit Frontend
Client/UI connecting to FastAPI backend over HTTP for real-time customer churn assessment.
Does NOT load model.joblib or import joblib directly.
"""

import os
import requests
import streamlit as st

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Configurable currency exchange rate proxy (USD to INR)
USD_TO_INR_RATE = 90.0

st.set_page_config(
    page_title="Telco Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# CURRENCY & STRING FORMATTING HELPERS
# =====================================================================
def format_inr(amount: float) -> str:
    """
    Formats a numeric amount into standard Indian Rupee numbering format
    (e.g., 125000 -> ₹1,25,000).
    """
    amt_int = int(round(amount))
    s = str(amt_int)
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    remaining = s[:-3]
    groups = []
    while remaining:
        groups.append(remaining[-2:])
        remaining = remaining[:-2]
    groups.reverse()
    return f"₹{','.join(groups)},{last_three}"


def parse_driver_text(driver_str: str, default_desc: str = "Model-supported factor") -> tuple[str, str]:
    """
    Splits driver strings into a primary title and secondary context description.
    e.g. 'Short customer tenure (12 months)' -> ('Short customer tenure', '12 months with the company')
    """
    if "(" in driver_str and driver_str.endswith(")"):
        idx = driver_str.find("(")
        title = driver_str[:idx].strip()
        detail = driver_str[idx + 1 : -1].strip()
        if detail.endswith("months"):
            detail = f"{detail} with the company"
        elif detail.startswith("$"):
            detail = f"Current billing rate: {detail}"
        return title, detail
    elif ":" in driver_str:
        parts = driver_str.split(":", 1)
        return parts[0].strip(), parts[1].strip()
    return driver_str, default_desc


# =====================================================================
# CUSTOM CSS STYLING
# =====================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }

    /* Result Dashboard Cards */
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        min-height: 135px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }

    .metric-value-primary {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }

    .metric-value-exposure {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }

    .metric-subtext {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 0.3rem;
    }

    /* Risk Badges */
    .badge-high {
        color: #DC2626;
        font-weight: 700;
    }
    .badge-medium {
        color: #D97706;
        font-weight: 700;
    }
    .badge-low {
        color: #16A34A;
        font-weight: 700;
    }

    /* Driver List Cards */
    .driver-item {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.65rem;
    }

    .driver-badge {
        background: #E2E8F0;
        color: #334155;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 6px;
        min-width: 36px;
        text-align: center;
    }

    .driver-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.15rem;
    }

    .driver-detail {
        font-size: 0.85rem;
        color: #64748B;
    }

    /* Protective Factor List Cards */
    .protective-item {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        background: #F0FDF4;
        border: 1px solid #DCFCE7;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.65rem;
    }

    .protective-badge {
        background: #DCFCE7;
        color: #166534;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 6px;
        min-width: 36px;
        text-align: center;
    }

    /* Form Section Dividers */
    .group-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.6rem;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #F1F5F9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# BACKEND STATUS CHECK (FOR SIDEBAR)
# =====================================================================
backend_connected = False
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=2)
    if health_resp.status_code == 200:
        backend_connected = True
except requests.exceptions.RequestException:
    backend_connected = False


# =====================================================================
# HEADER SECTION (CLEAN & NON-TECHNICAL)
# =====================================================================
st.markdown('<div class="main-title">Telco Customer Churn Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Assess customer churn risk and understand potential revenue exposure using our trained machine learning model.</div>',
    unsafe_allow_html=True,
)


# =====================================================================
# SIDEBAR: QUICK CUSTOMER PROFILES
# =====================================================================
with st.sidebar:
    st.subheader("Quick Customer Profiles")
    st.caption("Try a sample profile to explore the model.")

    preset = st.radio(
        "Choose Profile:",
        ["Custom Input", "High-Risk Customer", "Low-Risk Customer"],
        index=0,
    )

    st.divider()

    st.markdown("##### System Status")
    if backend_connected:
        st.markdown(
            """
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.6;">
                <div><strong>Backend:</strong> <span style="color: #16A34A; font-weight: 600;">● Healthy</span></div>
                <div><strong>Model:</strong> <code>telco_churn_model</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="font-size: 0.85rem; color: #DC2626; line-height: 1.6;">
                <div><strong>Backend:</strong> <strong>● Offline</strong></div>
                <div style="color: #64748B; margin-top: 4px;">Start with: <code>uvicorn api.main:app</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================================
# PRESET PROFILE VALUES
# =====================================================================
if preset == "High-Risk Customer":
    def_gender = "Female"
    def_senior = 0
    def_partner = "No"
    def_dependents = "No"
    def_tenure = 2
    def_phone = "Yes"
    def_lines = "Yes"
    def_internet = "Fiber optic"
    def_sec = "No"
    def_backup = "No"
    def_device = "No"
    def_tech = "No"
    def_tv = "Yes"
    def_movies = "No"
    def_contract = "Month-to-month"
    def_paperless = "Yes"
    def_payment = "Electronic check"
    def_monthly = 85.50
    def_total = 171.00
elif preset == "Low-Risk Customer":
    def_gender = "Male"
    def_senior = 0
    def_partner = "Yes"
    def_dependents = "Yes"
    def_tenure = 68
    def_phone = "Yes"
    def_lines = "Yes"
    def_internet = "DSL"
    def_sec = "Yes"
    def_backup = "Yes"
    def_device = "Yes"
    def_tech = "Yes"
    def_tv = "Yes"
    def_movies = "Yes"
    def_contract = "Two year"
    def_paperless = "No"
    def_payment = "Bank transfer (automatic)"
    def_monthly = 84.80
    def_total = 5766.40
else:
    def_gender = "Male"
    def_senior = 0
    def_partner = "No"
    def_dependents = "Yes"
    def_tenure = 60
    def_phone = "Yes"
    def_lines = "Yes"
    def_internet = "Fiber optic"
    def_sec = "Yes"
    def_backup = "Yes"
    def_device = "Yes"
    def_tech = "Yes"
    def_tv = "Yes"
    def_movies = "Yes"
    def_contract = "Two year"
    def_paperless = "No"
    def_payment = "Bank transfer (automatic)"
    def_monthly = 85.00
    def_total = 5100.00


# =====================================================================
# CUSTOMER INPUT SECTION
# =====================================================================
st.markdown("### Customer Profile")
st.caption("Enter the customer's current information to assess their churn risk.")

with st.form("customer_input_form"):
    col_group1, col_group2, col_group3 = st.columns(3)

    # GROUP 1: Customer Profile
    with col_group1:
        st.markdown('<div class="group-header">Customer Profile</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(def_gender))
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=[0, 1].index(def_senior),
        )
        partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(def_partner))
        dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(def_dependents))
        tenure = st.number_input("Tenure (months with company)", min_value=0, max_value=72, value=def_tenure, step=1)

    # GROUP 2: Services
    with col_group2:
        st.markdown('<div class="group-header">Services</div>', unsafe_allow_html=True)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(def_phone))
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(def_lines))
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(def_internet))
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_sec))
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_backup))
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_device))
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_tech))
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_tv))
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_movies))

    # GROUP 3: Billing & Contract
    with col_group3:
        st.markdown('<div class="group-header">Billing & Contract</div>', unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(def_contract))
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(def_paperless))
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(def_payment),
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=float(def_monthly), step=0.50, format="%.2f")
        total_charges = st.number_input("Total Charges ($ to date)", min_value=0.0, max_value=12000.0, value=float(def_total), step=10.0, format="%.2f")

    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1.2, 1])
    with btn_col2:
        submit_button = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)
        st.caption("<div style='text-align: center; color: #64748B;'>The customer profile will be sent to the prediction service.</div>", unsafe_allow_html=True)


# =====================================================================
# RESULTS DASHBOARD
# =====================================================================
if submit_button:
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        with st.spinner("Analyzing customer risk profile..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()

            churn_pred = result.get("churn_prediction", "No")
            churn_prob = float(result.get("churn_probability", 0.0))
            raw_risk_level = result.get("risk_level", "Low")
            revenue_exposure_usd = float(result.get("estimated_revenue_exposure", 0.0))
            risk_drivers = result.get("key_risk_drivers", [])
            protective_factors = result.get("protective_factors", [])

            # Compute INR revenue exposure proxy
            revenue_exposure_inr = revenue_exposure_usd * USD_TO_INR_RATE
            exposure_inr_formatted = format_inr(revenue_exposure_inr)

            # Customer-friendly prediction copy
            if churn_pred == "Yes":
                prediction_text = "Likely to leave"
            else:
                prediction_text = "Likely to stay"

            # Risk level badge styling
            if raw_risk_level == "High":
                risk_label = "High Churn Risk"
                risk_class = "badge-high"
            elif raw_risk_level == "Medium":
                risk_label = "Medium Churn Risk"
                risk_class = "badge-medium"
            else:
                risk_label = "Low Churn Risk"
                risk_class = "badge-low"

            st.divider()
            st.markdown("### Customer Churn Risk Assessment")

            # 4 Metric Cards Layout (Card 4 prominently displays the INR revenue exposure)
            rc1, rc2, rc3, rc4 = st.columns(4)

            with rc1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Prediction</div>
                        <div class="metric-value-primary">{prediction_text}</div>
                        <div class="metric-subtext">Customer status outcome</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with rc2:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Churn Probability</div>
                        <div class="metric-value-primary">{churn_prob * 100:.1f}%</div>
                        <div class="metric-subtext">Statistical likelihood</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with rc3:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Risk Level</div>
                        <div class="metric-value-primary {risk_class}">{risk_label}</div>
                        <div class="metric-subtext">Operational tier</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with rc4:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Estimated Revenue Exposure</div>
                        <div class="metric-value-exposure">{exposure_inr_formatted} <span style="font-size: 0.9rem; font-weight: 500; color: #64748B;">/ year</span></div>
                        <div class="metric-subtext">Annualized probability-weighted proxy</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

            # Visual Churn Probability Indicator
            st.markdown(f"**Churn Probability:** `{churn_prob * 100:.1f}%`")
            st.progress(min(max(churn_prob, 0.0), 1.0))
            
            pb_c1, pb_c2, pb_c3 = st.columns([0.4, 0.3, 0.3])
            with pb_c1:
                st.caption("0% — Low Risk (< 40%)")
            with pb_c2:
                st.caption("40% — Medium Risk (40% to 70%)")
            with pb_c3:
                st.caption("70% — High Risk (> 70%)")

            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

            # Key Risk Drivers
            st.markdown("#### Why might this customer leave?")
            st.caption("These factors are supported by the current model's interpretation.")

            is_no_drivers = (
                not risk_drivers
                or (len(risk_drivers) == 1 and "No major model-supported" in risk_drivers[0])
            )

            if is_no_drivers:
                st.info("No major model-supported churn drivers identified for this profile.")
            else:
                for idx, driver in enumerate(risk_drivers, 1):
                    driver_title, driver_detail = parse_driver_text(driver, default_desc="Model-supported risk factor")
                    st.markdown(
                        f"""
                        <div class="driver-item">
                            <div class="driver-badge">{idx:02d}</div>
                            <div>
                                <div class="driver-title">{driver_title}</div>
                                <div class="driver-detail">{driver_detail}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Factors reducing churn risk (Protective Factors)
            if protective_factors:
                st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
                st.markdown("#### Factors reducing churn risk")
                st.caption("These attributes are associated with lower churn risk for this customer profile.")

                for idx, factor in enumerate(protective_factors, 1):
                    factor_title, factor_detail = parse_driver_text(factor, default_desc="Model-supported retention factor")
                    st.markdown(
                        f"""
                        <div class="protective-item">
                            <div class="protective-badge">{idx:02d}</div>
                            <div>
                                <div class="driver-title">{factor_title}</div>
                                <div class="driver-detail">{factor_detail}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)

            # Expandable Revenue Exposure Explanation (Explains the figure shown in Card 4 above)
            with st.expander("How is Estimated Revenue Exposure calculated?"):
                st.markdown(
                    f"""
                    This section explains how the **{exposure_inr_formatted} / year** Estimated Revenue Exposure shown above is calculated.

                    **Definition:**  
                    Estimated Revenue Exposure is a probability-weighted annual revenue proxy based on the customer's current monthly charge, assuming 12 months of potential future billing. It is not a separate ML prediction of actual revenue loss.

                    **Calculation & Currency Conversion Assumption:**  
                    Revenue Exposure is calculated using the customer's current monthly charge, churn probability, 12 months of potential billing, and the configured USD-to-INR conversion rate (`USD_TO_INR_RATE = {USD_TO_INR_RATE}`):

                    $$\\text{{Estimated Revenue Exposure (INR)}} = P(\\text{{Churn}}) \\times \\text{{Monthly Charges (USD)}} \\times 12 \\times {USD_TO_INR_RATE}$$
                    """
                )

        elif response.status_code == 422:
            st.error(f"Input validation error from API: {response.text}")
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ **Prediction API is unavailable.**\n\n"
            f"Could not connect to FastAPI at `{API_URL}`. Please make sure the backend server is running:\n"
            f"```bash\nuvicorn api.main:app --reload --port 8000\n```"
        )
    except requests.exceptions.Timeout:
        st.error("⏱️ Request to FastAPI backend timed out. Please try again.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
