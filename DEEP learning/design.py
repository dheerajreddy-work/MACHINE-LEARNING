
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 50%,
        #0f172a 100%
    );
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ================= TITLE ================= */

.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 35px;
}


/* ================= SECTION TITLE ================= */

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #38bdf8;
    margin-top: 10px;
    margin-bottom: 20px;
}


/* ================= RESULT CARDS ================= */

.churn-card {
    background: rgba(127, 29, 29, 0.85);
    border: 2px solid #ef4444;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
}

.safe-card {
    background: rgba(20, 83, 45, 0.85);
    border: 2px solid #22c55e;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
}

.result-title {
    font-size: 26px;
    font-weight: 700;
    color: white;
    margin-bottom: 15px;
}

.probability {
    font-size: 48px;
    font-weight: 800;
    color: #38bdf8;
    margin: 10px 0;
}

.probability-label {
    font-size: 16px;
    color: #cbd5e1;
}


/* ================= BUTTON ================= */

.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        90deg,
        #0284c7,
        #38bdf8
    );
    color: white;
    font-size: 19px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(
        90deg,
        #0369a1,
        #0ea5e9
    );
}


/* ================= SIDEBAR ================= */

[data-testid="stSidebar"] {
    background: #020617;
}

[data-testid="stSidebar"] p {
    color: #cbd5e1;
}


/* ================= INPUT LABELS ================= */

label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}


/* ================= DIVIDER ================= */

hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model("model.h5")

    return model


# ============================================================
# LOAD ENCODERS AND SCALER
# ============================================================

@st.cache_resource
def load_preprocessors():

    with open("onehot_encoder_geo.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)

    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("scalar.pkl", "rb") as file:
        scalar = pickle.load(file)

    return (
        onehot_encoder_geo,
        label_encoder_gender,
        scalar
    )


# Load model

model = load_model()


# Load preprocessors

(
    onehot_encoder_geo,
    label_encoder_gender,
    scalar
) = load_preprocessors()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 Churn AI")

    st.markdown("---")

    st.markdown("""
### About

This application uses a trained
**Artificial Neural Network (ANN)**
to predict whether a customer is
likely to churn.

### Model Pipeline

📊 Customer Data

↓

🔤 Encoding

↓

📏 Standard Scaling

↓

🧠 Neural Network

↓

🎯 Churn Prediction
""")

    st.markdown("---")

    st.markdown("### 🎯 Prediction Rule")

    st.info("""
**Probability > 50%**

🔴 Customer likely to churn


**Probability ≤ 50%**

🟢 Customer not likely to churn
""")

    st.markdown("---")

    st.caption("Customer Churn Prediction System")


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered customer retention analysis</div>',
    unsafe_allow_html=True
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    geography = st.selectbox(
        "🌍 Geography",
        onehot_encoder_geo.categories_[0]
    )

    gender = st.selectbox(
        "⚧ Gender",
        label_encoder_gender.classes_
    )

    age = st.slider(
        "🎂 Age",
        min_value=18,
        max_value=92,
        value=35
    )

    tenure = st.slider(
        "📅 Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    num_of_products = st.slider(
        "📦 Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    credit_score = st.number_input(
        "💳 Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    balance = st.number_input(
        "💰 Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    estimated_salary = st.number_input(
        "💵 Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    has_cr_card = st.selectbox(
        "💳 Has Credit Card?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_active_member = st.selectbox(
        "👤 Is Active Member?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Customer Churn"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Prepare Input Data
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],

        "Gender": [
            label_encoder_gender.transform([gender])[0]
        ],

        "Age": [age],

        "Tenure": [tenure],

        "Balance": [balance],

        "NumOfProducts": [num_of_products],

        "HasCrCard": [has_cr_card],

        "IsActiveMember": [is_active_member],

        "EstimatedSalary": [estimated_salary]
    })


    # --------------------------------------------------------
    # One Hot Encode Geography
    # --------------------------------------------------------

    geo_encoded = onehot_encoder_geo.transform(
        [[geography]]
    ).toarray()


    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(
            ["Geography"]
        )
    )


    # --------------------------------------------------------
    # Combine Input Data
    # --------------------------------------------------------

    input_data = pd.concat(
        [
            input_data.reset_index(drop=True),
            geo_encoded_df
        ],
        axis=1
    )


    # --------------------------------------------------------
    # Scale Input Data
    # --------------------------------------------------------

    input_scaled = scalar.transform(
        input_data
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_scaled,
        verbose=0
    )


    pred_prob = float(prediction[0][0])

    probability = pred_prob * 100


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # HIGH CHURN RISK
    # ========================================================

    if pred_prob > 0.5:

        result_html = (
            '<div class="churn-card">'
            '<div class="result-title">'
            '⚠️ Customer is likely to churn'
            '</div>'
            '<div class="probability">'
            f'{probability:.2f}%'
            '</div>'
            '<div class="probability-label">'
            'Churn Probability'
            '</div>'
            '</div>'
        )

        st.markdown(
            result_html,
            unsafe_allow_html=True
        )

        st.warning(
            "⚠️ Consider taking customer retention actions."
        )


    # ========================================================
    # LOW CHURN RISK
    # ========================================================

    else:

        result_html = (
            '<div class="safe-card">'
            '<div class="result-title">'
            '✅ Customer is not likely to churn'
            '</div>'
            '<div class="probability">'
            f'{probability:.2f}%'
            '</div>'
            '<div class="probability-label">'
            'Churn Probability'
            '</div>'
            '</div>'
        )

        st.markdown(
            result_html,
            unsafe_allow_html=True
        )

        st.success(
            "✅ Customer appears to have a low churn risk."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div style="text-align:center; color:#64748b; '
    'font-size:14px; padding:10px;">'
    '🤖 Customer Churn Prediction | '
    'Powered by TensorFlow & Streamlit'
    '</div>',
    unsafe_allow_html=True
)