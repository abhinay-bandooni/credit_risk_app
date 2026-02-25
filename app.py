import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Risk Engine", layout="wide")

st.title("💳 Credit Risk Scoring Engine")
st.markdown("### AI-powered Credit Default Prediction System")

@st.cache_resource
def load_model():
    return joblib.load("credit_risk_pipeline.pkl")

pipeline = load_model()
st.sidebar.header("Applicant Information")

LIMIT_BAL = st.sidebar.number_input("Credit Limit", min_value=0)
AGE = st.sidebar.number_input("Age", min_value=18, max_value=100)
SEX = st.sidebar.selectbox("Gender", ["Male", "Female"])
EDUCATION = st.sidebar.selectbox("Education Level", ["Graduate", "University", "High School"])
MARRIAGE = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Other"])
pay_0 = st.sidebar.number_input("Payment Status in April:", min_value=-2, max_value=10)
pay_2 = st.sidebar.number_input("Payment Status in May:", min_value=-2, max_value=10)
pay_3 = st.sidebar.number_input("Payment Status in June:", min_value=-2, max_value=10)
pay_4 = st.sidebar.number_input("Payment Status in July:", min_value=-2, max_value=10)
pay_5 = st.sidebar.number_input("Payment Status in Aug:", min_value=-2, max_value=10)
pay_6 = st.sidebar.number_input("Payment Status in Sep:", min_value=-2, max_value=10)

bill_amt1 = st.sidebar.number_input("Bill amount in April:")
bill_amt2 = st.sidebar.number_input("Bill amount in May:")
bill_amt3 = st.sidebar.number_input("Bill amount in June:")
bill_amt4 = st.sidebar.number_input("Bill amount in July:")
bill_amt5 = st.sidebar.number_input("Bill amount in Aug:")
bill_amt6 = st.sidebar.number_input("Bill amount in Sep:")

pay_amt1 = st.sidebar.number_input("Payed amount in April:")
pay_amt2 = st.sidebar.number_input("Payed amount in May:")
pay_amt3 = st.sidebar.number_input("Payed amount in June:")
pay_amt4 = st.sidebar.number_input("Payed amount in July:")
pay_amt5 = st.sidebar.number_input("Payed amount in Aug:")
pay_amt6 = st.sidebar.number_input("Payed amount in Sep:")

if SEX=="Male":
    SEX=1
else:
    SEX=2

if EDUCATION=="Graduate":
    EDUCATION=1
elif EDUCATION=="University":
    EDUCATION=2
elif EDUCATION=="High School":
    EDUCATION=3
else:
    EDUCATION=0

if MARRIAGE=="Married":
    MARRIAGE=1
elif MARRIAGE=='Single':
    MARRIAGE=2
else:
    MARRIAGE=3

if st.sidebar.button("Evaluate Risk"):

    input_data = pd.DataFrame([{
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": SEX,
        "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,
        "AGE": AGE,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "BILL_AMT1":bill_amt1,
        "BILL_AMT2":bill_amt2,
        "BILL_AMT3":bill_amt3,
        "BILL_AMT4":bill_amt4,
        "BILL_AMT5":bill_amt5,
        "BILL_AMT6":bill_amt6,
        "PAY_AMT1":pay_amt1,
        "PAY_AMT2":pay_amt2,
        "PAY_AMT3":pay_amt3,
        "PAY_AMT4":pay_amt4,
        "PAY_AMT5":pay_amt5,
        "PAY_AMT6":pay_amt6
    }])

    prediction = pipeline.predict(input_data)
    probability = pipeline.predict_proba(input_data)[0][1]

    st.subheader("Risk Assessment Result")

    if prediction == 1:
        st.error(f"⚠️ High Default Risk")
    else:
        st.success(f"✅ Low Default Risk")

    st.metric("Default Probability", f"{probability:.2%}")

st.markdown("---")
st.subheader("Business Interpretation")

st.write("""
This model helps financial institutions:
- Reduce default rates
- Assign risk-based interest rates
- Improve portfolio risk management
""")