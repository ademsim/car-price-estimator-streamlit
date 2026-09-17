import json
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Used Car Price Prediction", page_icon="🚗", layout="centered"
)

# app.py dosyasının bulunduğu klasörün yolunu otomatik alıyoruz
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Dosyaları Yükleme (Önbelleğe alarak performans artırıyoruz)
@st.cache_resource
def load_artifacts():
  model_path = os.path.join(BASE_DIR, "car_model.pkl")
  preprocessor_path = os.path.join(BASE_DIR, "car_preprocessor.pkl")
  columns_path = os.path.join(BASE_DIR, "columns.json")

  model = pickle.load(open(model_path, "rb"))
  preprocessor = pickle.load(open(preprocessor_path, "rb"))

  with open(columns_path, "r", encoding="utf-8") as f:
    columns = json.load(f)

  return model, preprocessor, columns


# Dosyaları yükleyip değişkenlere atıyoruz
try:
  model, preprocessor, expected_columns = load_artifacts()
except Exception as e:
  st.error(
      f"Model veya dosya yüklenirken hata oluştu! Lütfen pkl ve json"
      f" dosyalarının dizinde olduğundan emin olun. Hata: {e}"
  )
  st.stop()


st.title("🚗 İkinci El Araç Fiyat Tahmin Uygulaması")
st.markdown(
    "Araç özelliklerini girerek tahmini piyasa fiyatını anında öğrenin."
)
st.divider()

# Kullanıcı Girdi Alanları (Form Yapısı)
with st.form("prediction_form"):
  col1, col2 = st.columns(2)

  with col1:
    year = st.number_input(
        "Üretim Yılı", min_value=1990, max_value=2026, value=2018
    )
    km_driven = st.number_input(
        "Kilometre (KM)", min_value=0, max_value=500000, value=50000, step=1000
    )
    engine = st.number_input(
        "Motor Hacmi (CC)", min_value=500, max_value=5000, value=1200
    )
    max_power = st.number_input(
        "Maksimum Güç (bhp)", min_value=30, max_value=500, value=85
    )

  with col2:
    seats = st.selectbox("Koltuk Sayısı", [2, 4, 5, 6, 7, 8, 9, 10], index=2)
    fuel = st.selectbox("Yakıt Tipi", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox(
        "Satıcı Türü", ["Individual", "Dealer", "Trustmark Dealer"]
    )
    transmission = st.selectbox("Vites Tipi", ["Manual", "Automatic"])
    owner = st.selectbox(
        "Kaçıncı Sahibinden",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car",
        ],
    )

  # Tahmin Butonu
  submit_button = st.form_submit_button(
      label="🔮 Fiyat Tahmini Yap", use_container_width=True
  )

# Tahmin Hesaplama Mantığı
if submit_button:
  # Kullanıcı verilerini modelin beklediği sütun yapısına uygun bir DataFrame'e dönüştürüyoruz
  input_data = pd.DataFrame(
      0, index=[0], columns=expected_columns
  )  # Varsayılan 0 doldurma

  # Sayısal değerleri ata
  if "year" in input_data.columns:
    input_data["year"] = year
  if "km_driven" in input_data.columns:
    input_data["km_driven"] = km_driven
  if "engine" in input_data.columns:
    input_data["engine"] = engine
  if "max_power" in input_data.columns:
    input_data["max_power"] = max_power
  if "seats" in input_data.columns:
    input_data["seats"] = seats

  # Kategorik eşlemeler veya dummy sütunlar için temel atamalar
  for col in input_data.columns:
    if fuel.lower() in col.lower() and col in input_data.columns:
      input_data[col] = 1
    if seller_type.lower() in col.lower() and col in input_data.columns:
      input_data[col] = 1
    if transmission.lower() in col.lower() and col in input_data.columns:
      input_data[col] = 1
    if owner.lower() in col.lower() and col in input_data.columns:
      input_data[col] = 1

  try:
    # Preprocessor ile dönüştür ve tahmin et
    processed_input = preprocessor.transform(input_data)
    prediction = model.predict(processed_input)[0]

    # Sonucu ekrana yazdır
    st.success(f"🎉 Tahmini Araç Fiyatı: **{prediction:,.2f} TL**")
  except Exception as e:
    st.error(
        "Tahmin yapılırken bir dönüştürme hatası oluştu. Lütfen kolon"
        f" uyumluluğunu kontrol edin. Hata: {e}"
    )
