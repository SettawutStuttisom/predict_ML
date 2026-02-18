import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ===============================
# โหลดโมเดลและ scaler
# ===============================
model = joblib.load("xgb_model (1).pkl")
scaler = joblib.load("scaler (1).pkl")

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
    "Graduate School (บัณฑิตศึกษา)": 1,
    "University (ปริญญาตรี)": 2,
    "High School (มัธยมศึกษา)": 3,
    "Others (อื่น ๆ / ไม่ระบุ)": 4
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
st.subheader("📌 ประวัติการชำระย้อนหลัง 6 เดือน")
st.caption("เลือกสถานะการชำระในแต่ละเดือน")

pay_options = ["จ่ายครบ", "จ่ายตรงเวลา", "ค้างชำระ"]

pay_values = []

for i, label in enumerate([
    "เดือนล่าสุด",
    "2 เดือนก่อน",
    "3 เดือนก่อน",
    "4 เดือนก่อน",
    "5 เดือนก่อน",
    "6 เดือนก่อน"
]):
    status = st.selectbox(label, pay_options, key=i)
    pay_values.append(status)

# นับจำนวนเดือนที่ค้าง
late_count = sum(1 for p in pay_values if p == "ค้างชำระ")

# จำกัดค่าสูงสุดที่ 2 (ตาม dataset)
late_level = min(late_count, 2)

# แปลงค่าเข้าโมเดล
def convert_status(status):
    if status == "จ่ายครบ":
        return -1
    elif status == "จ่ายตรงเวลา":
        return 0
    elif status == "ค้างชำระ":
        return late_level

PAY_0 = convert_status(pay_values[0])
PAY_2 = convert_status(pay_values[1])
PAY_3 = convert_status(pay_values[2])
PAY_4 = convert_status(pay_values[3])
PAY_5 = convert_status(pay_values[4])
PAY_6 = convert_status(pay_values[5])

st.info(f"จำนวนเดือนที่ค้างสะสม: {late_count} เดือน")
st.write(f"ระดับความรุนแรงที่ส่งเข้าโมเดล: {late_level}")

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
    st.write("ความน่าจะเป็นของการผิดนัดชำระหนี้:",
         str(round(probability * 100, 2)) + "%")

    if prediction[0] == 1:
        st.error("⚠️ ลูกค้ามีความเสี่ยงผิดนัดชำระ")
    else:
        st.success("✅ ลูกค้าไม่น่าจะผิดนัดชำระ")
        
st.markdown("---")
st.subheader("📈 ประสิทธิภาพของโมเดล (Test Set)")

st.write("Accuracy:", "82%")
st.write("ROC-AUC:", "0.773")
st.write("F1-score (กลุ่มเสี่ยง):", "0.47")

st.caption("ค่าประสิทธิภาพวัดจากชุดข้อมูลทดสอบจำนวน 6,000 ตัวอย่าง")