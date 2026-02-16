import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ===============================
# โหลดโมเดลและ scaler
# ===============================
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Default Prediction", page_icon="💳")

st.title("💳 Credit Default Prediction")
st.markdown("ระบบทำนายความเสี่ยงการผิดนัดชำระหนี้บัตรเครดิตด้วย XGBoost")

st.markdown("---")

# ===============================
# รับค่า input
# ===============================

st.subheader("📌 ข้อมูลพื้นฐานลูกค้า")

LIMIT_BAL = st.number_input(
    "LIMIT_BAL (วงเงินเครดิต)",
    min_value=0,
    value=20000,
    help="วงเงินเครดิตที่ธนาคารอนุมัติให้ลูกค้า"
)

# ✅ เปลี่ยน EDUCATION เป็น selectbox พร้อมคำอธิบาย
education_dict = {
    "1 = Graduate School (บัณฑิตศึกษา)": 1,
    "2 = University (ปริญญาตรี)": 2,
    "3 = High School (มัธยมศึกษา)": 3,
    "4 = Others (อื่น ๆ / ไม่ระบุ)": 4
}

education_label = st.selectbox(
    "EDUCATION (ระดับการศึกษา)",
    list(education_dict.keys()),
    help="ระดับการศึกษาของลูกค้า ซึ่งมีผลต่อความสามารถในการชำระหนี้"
)

EDUCATION = education_dict[education_label]

AGE = st.number_input(
    "AGE (อายุ)",
    min_value=18,
    max_value=100,
    value=25
)

st.markdown("---")
st.subheader("📌 ประวัติการชำระย้อนหลัง (-1 ถึง 6)")
st.caption("-1 = จ่ายครบ | 0 = จ่ายตรงเวลา | 1-6 = ค้างชำระ X เดือน")

PAY_0 = st.number_input("PAY_0 (เดือนล่าสุด)", min_value=-1, max_value=6, value=0, step=1)
PAY_2 = st.number_input("PAY_2 (2 เดือนก่อน)", min_value=-1, max_value=6, value=0, step=1)
PAY_3 = st.number_input("PAY_3 (3 เดือนก่อน)", min_value=-1, max_value=6, value=0, step=1)
PAY_4 = st.number_input("PAY_4 (4 เดือนก่อน)", min_value=-1, max_value=6, value=0, step=1)
PAY_5 = st.number_input("PAY_5 (5 เดือนก่อน)", min_value=-1, max_value=6, value=0, step=1)
PAY_6 = st.number_input("PAY_6 (6 เดือนก่อน)", min_value=-1, max_value=6, value=0, step=1)

st.markdown("---")
st.subheader("📌 ข้อมูลทางการเงิน")

BILL_AMT1 = st.number_input(
    "BILL_AMT1 (ยอดค้างชำระล่าสุด)",
    min_value=0,
    value=5000
)

PAY_AMT3 = st.number_input(
    "PAY_AMT3 (จำนวนเงินที่จ่ายเดือนที่ 3)",
    min_value=0,
    value=1000
)

# ===============================
# รวมเป็น DataFrame
# ===============================

input_data = pd.DataFrame([[ 
    LIMIT_BAL,
    EDUCATION,
    AGE,
    PAY_0,
    PAY_2,
    PAY_3,
    PAY_4,
    PAY_5,
    PAY_6,
    BILL_AMT1,
    PAY_AMT3
]], columns=[
    "LIMIT_BAL",
    "EDUCATION",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "PAY_AMT3"
])

# ===============================
# Predict
# ===============================

st.markdown("---")

if st.button("🔍 Predict"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 ผลการทำนาย")
    st.write("Probability of Default:", round(probability, 4))

    if prediction[0] == 1:
        st.error("⚠️ ลูกค้ามีความเสี่ยงผิดนัดชำระ")
    else:
        st.success("✅ ลูกค้าไม่น่าจะผิดนัดชำระ")
