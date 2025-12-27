import streamlit as st
import pickle
import numpy as np

# =====================
# LOAD MODEL
# =====================
with open("heart_disease_stacking_optuna.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🫀 Aplikasi Prediksi Risiko Penyakit Jantung")
st.write("Model: **Ensemble Stacking (Optuna + SMOTE)**")
st.markdown("---")

st.header("📌 Data Pasien")

# =====================
# INPUT DATA
# =====================
bmi = st.number_input("BMI (Indeks Massa Tubuh)", 10.0, 50.0, 25.0)

smoking = st.selectbox("Apakah Merokok?", ["Tidak", "Ya"])
smoking = 1 if smoking == "Ya" else 0

alcohol = st.selectbox("Konsumsi Alkohol?", ["Tidak", "Ya"])
alcohol = 1 if alcohol == "Ya" else 0

stroke = st.selectbox("Pernah Stroke?", ["Tidak", "Ya"])
stroke = 1 if stroke == "Ya" else 0

physical_health = st.number_input(
    "Jumlah hari sakit fisik (0–30 hari)", 0, 30, 0
)

mental_health = st.number_input(
    "Jumlah hari gangguan kesehatan mental (0–30 hari)", 0, 30, 0
)

diff_walking = st.selectbox("Kesulitan Berjalan?", ["Tidak", "Ya"])
diff_walking = 1 if diff_walking == "Ya" else 0

sex = st.selectbox("Jenis Kelamin", ["Wanita", "Pria"])
sex = 1 if sex == "Pria" else 0

# =====================
# USIA JELAS
# =====================
age_label = st.selectbox(
    "Kategori Usia",
    [
        "18–24 tahun",
        "25–29 tahun",
        "30–34 tahun",
        "35–39 tahun",
        "40–44 tahun",
        "45 tahun ke atas"
    ]
)
age_map = {
    "18–24 tahun": 1,
    "25–29 tahun": 2,
    "30–34 tahun": 3,
    "35–39 tahun": 4,
    "40–44 tahun": 5,
    "45 tahun ke atas": 6
}
age = age_map[age_label]

# =====================
# RAS
# =====================
race_label = st.selectbox(
    "Ras",
    ["White", "Black", "Asian", "Hispanic", "Other"]
)
race_map = {
    "White": 1,
    "Black": 2,
    "Asian": 3,
    "Hispanic": 4,
    "Other": 5
}
race = race_map[race_label]

diabetic = st.selectbox("Diabetes?", ["Tidak", "Ya"])
diabetic = 1 if diabetic == "Ya" else 0

physical_activity = st.selectbox("Aktivitas Fisik Rutin?", ["Tidak", "Ya"])
physical_activity = 1 if physical_activity == "Ya" else 0

# =====================
# KESEHATAN UMUM
# =====================
genhealth_label = st.selectbox(
    "Kondisi Kesehatan Umum",
    ["Sangat Baik", "Baik", "Cukup", "Buruk", "Sangat Buruk"]
)
genhealth_map = {
    "Sangat Baik": 1,
    "Baik": 2,
    "Cukup": 3,
    "Buruk": 4,
    "Sangat Buruk": 5
}
genhealth = genhealth_map[genhealth_label]

sleep_time = st.number_input("Waktu Tidur per Hari (jam)", 1.0, 12.0, 8.0)

asthma = st.selectbox("Asma?", ["Tidak", "Ya"])
asthma = 1 if asthma == "Ya" else 0

kidney = st.selectbox("Penyakit Ginjal?", ["Tidak", "Ya"])
kidney = 1 if kidney == "Ya" else 0

skincancer = st.selectbox("Kanker Kulit?", ["Tidak", "Ya"])
skincancer = 1 if skincancer == "Ya" else 0

st.markdown("---")

# =====================
# PREDIKSI
# =====================
if st.button("🔍 Prediksi Risiko Penyakit Jantung"):
    input_data = np.array([[
        bmi, smoking, alcohol, stroke,
        physical_health, mental_health, diff_walking,
        sex, age, race, diabetic, physical_activity,
        genhealth, sleep_time, asthma, kidney, skincancer
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.error(
            f"⚠ **RISIKO TINGGI** terkena penyakit jantung\n\n"
            f"Probabilitas: **{probability:.2%}**"
        )
    else:
        st.success(
            f"✅ **RISIKO RENDAH** terkena penyakit jantung\n\n"
            f"Probabilitas: **{probability:.2%}**"
        )

    st.info(
        "⚠ Catatan: Hasil ini merupakan prediksi berbasis machine learning "
        "dan **bukan diagnosis medis resmi**."
    )
