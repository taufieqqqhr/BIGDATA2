import os
import random
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.linear_model import LinearRegression
import joblib

# 1. DEFINISI ABSOLUTE PATH (Wajib sesuai aturan soal)
BASE_DIR = "/home/asus/bigdata-project"
OUTPUT_TOTAL = os.path.join(BASE_DIR, "output/attendance_total")
OUTPUT_TIME = os.path.join(BASE_DIR, "output/attendance_time")
OUTPUT_ML = os.path.join(BASE_DIR, "output/ml_attendance")
MODEL_PATH = os.path.join(BASE_DIR, "models/lr_model.pkl")

print("=== LANGKAH 1: Generate Data Simulasi (100 Menit) ===")
# List Gedung sesuai instruksi soal
buildings = ["Fakultas Sains dan Teknologi", "Perpustakaan", "Auditorium"]
start_time = datetime.now()

# List untuk menampung data mentah
raw_data = []

# Generate data untuk setiap menit selama 100 menit ke depan
for minute_offset in range(100):
    current_timestamp = start_time + timedelta(minutes=minute_offset)
    for bld in buildings:
        # Angka acak 20 sampai 300 mahasiswa
        count = random.randint(20, 300)
        raw_data.append((current_timestamp, bld, count))

print(f"Berhasil men-generate {len(raw_data)} baris data simulasi.")


print("\n=== LANGKAH 2: Spark Processing & Transformation ===")
# Inisialisasi Spark Session
spark = SparkSession.builder \
    .appName("SmartCampusAttendance") \
    .getOrCreate()

# Definisikan schema kolom data
schema = ["timestamp", "building", "attendance_count"]

# Buat DataFrame awal dari data mentah
df = spark.createDataFrame(raw_data, schema=schema)

# Tambahkan kolom 'hour' dan pembulatan waktu per 20 menit untuk kebutuhan agregasi
df_processed = df.withColumn("hour", F.hour("timestamp")) \
                 .withColumn("minute_20", F.floor(F.minute("timestamp") / 20) * 20)

# 1. Total mahasiswa per gedung
df_total = df_processed.groupBy("building").agg(F.sum("attendance_count").alias("total_attendance"))

# 2. Tren kehadiran per 20 menit (Dikelompokkan berdasarkan jam dan blok 20 menitan)
df_time = df_processed.groupBy("building", "hour", "minute_20").agg(F.avg("attendance_count").alias("avg_attendance"))

# 3. Dataset AI berbasis jam (Agregasi rata-rata jumlah mahasiswa per jam untuk fitur ML)
df_ml_data = df_processed.groupBy("hour").agg(F.avg("attendance_count").alias("avg_attendance_count"))


print("\n=== LANGKAH 3: Menyimpan ke Format Parquet (Absolute Path) ===")
# Simpan hasil agregasi ke folder parquet masing-masing (di-overwrite agar aman jika running ulang)
df_total.write.mode("overwrite").parquet(OUTPUT_TOTAL)
df_time.write.mode("overwrite").parquet(OUTPUT_TIME)
df_ml_data.write.mode("overwrite").parquet(OUTPUT_ML)
print("Data Spark sukses disimpan ke dalam format Parquet.")


print("\n=== LANGKAH 4: Machine Learning Training (Linear Regression) ===")
# Ambil data dari Spark DataFrame ML ke Pandas untuk dilatih menggunakan Scikit-Learn
pandas_ml_df = df_ml_data.toPandas()

# Cek apakah data tersedia untuk di-training
if not pandas_ml_df.empty:
    X = pandas_ml_df[['hour']]             # Fitur: Jam
    y = pandas_ml_df['avg_attendance_count'] # Target: Jumlah mahasiswa

    # Inisialisasi dan latih model Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    # Simpan model yang sudah dilatih ke folder models/
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model AI Linear Regression berhasil dilatih dan disimpan!")
else:
    print("Gagal melatih model: Data kosong.")

# Hentikan Spark Session
spark.stop()
print("\n=== Pipeline Selesai Dijalankan! ===")