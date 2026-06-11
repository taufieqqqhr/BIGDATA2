import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# 1. KONFIGURASI PATH ABSOLUTE
BASE_DIR = "/home/asus/bigdata-project"
PATH_TOTAL = os.path.join(BASE_DIR, "output/attendance_total")
PATH_TIME = os.path.join(BASE_DIR, "output/attendance_time")
MODEL_PATH = os.path.join(BASE_DIR, "models/lr_model.pkl")

# Set Page Title
st.set_page_config(page_title="Smart Campus Attendance", layout="wide")
st.title("🏛️ Smart Campus Attendance Analytics")
st.write("Sistem Monitoring Kepadatan Mahasiswa di Berbagai Gedung")

# 2. MEMBACA DATA PARQUET (Menggunakan Pandas karena Spark tidak efisien untuk aplikasi web interaktif)
@st.cache_data
def load_data():
    df_total = pd.read_parquet(PATH_TOTAL)
    df_time = pd.read_parquet(PATH_TIME)
    return df_total, df_time

try:
    df_total, df_time = load_data()
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(
        f"Gagal memuat data/model. Pastikan pipeline_uas.py sudah dijalankan. Error: {e}"
    )
    st.stop()

# 3. SIDEBAR FILTER GEDUNG (Fitur Wajib)
st.sidebar.header("Filter Navigasi")
list_gedung = df_total["building"].unique().tolist()
selected_building = st.sidebar.selectbox("Pilih Gedung:", list_gedung)

# Filter tren data berdasarkan gedung yang dipilih
df_time_filtered = df_time[df_time["building"] == selected_building].sort_values(
    by=["hour", "minute_20"]
)

# 4. KPI TOTAL MAHASISWA (Fitur Wajib)
# Mengambil total keseluruhan mahasiswa di gedung yang dipilih
total_mhs_gedung = df_total[df_total["building"] == selected_building][
    "total_attendance"
].values[0]

st.subheader(f"Analisis Kepadatan: {selected_building}")
col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        label=f"Total Mahasiswa Masuk (100 Menit)",
        value=f"{int(total_mhs_gedung):,}",
    )
    st.caption("Data diperbarui secara berkala melalui tapping kartu mahasiswa.")

# 5. GRAFIK TREN KEHADIRAN PLOTLY (Fitur Wajib)
with col2:
    # Membuat kolom string penanda waktu untuk sumbu X grafik
    df_time_filtered["waktu_blok"] = (
        df_time_filtered["hour"].astype(str)
        + ":"
        + df_time_filtered["minute_20"].astype(str).str.zfill(2)
    )

    fig = px.line(
        df_time_filtered,
        x="waktu_blok",
        y="avg_attendance",
        title=f"Tren Kehadiran per 20 Menit di {selected_building}",
        labels={
            "waktu_blok": "Jam & Menit (Blok 20m)",
            "avg_attendance": "Rata-rata Mahasiswa",
        },
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 6. PREDIKSI KEPADATAN KAMPUS BERBASIS AI (Fitur Wajib)
st.subheader("🔮 AI Prediction (Linear Regression)")
st.write("Prediksi rata-rata kepadatan mahasiswa kampus berdasarkan jam sibuk.")

# Slider input jam dari user untuk prediksi AI
input_hour = st.slider(
    "Geser untuk memilih jam operasional kuliah (0 - 23):",
    min_value=0,
    max_value=23,
    value=10,
)

# Melakukan prediksi menggunakan model Linear Regression yang sudah dilatih
predicted_count = model.predict([[input_hour]])[0]
# Menghindari hasil prediksi minus jika tren menurun drastis
predicted_count = max(0, predicted_count)

st.success(
    f"Hasil Prediksi AI: Pada jam **{input_hour}:00**, rata-rata kepadatan di kampus diperkirakan sekitar **{int(predicted_count)}** mahasiswa."
)