# 🎓 JurusanFinder – Rekomendasi Jurusan Kuliah Pintar

> Sistem rekomendasi jurusan kuliah berbasis AI yang membantu siswa menentukan pilihan jurusan optimal berdasarkan kepribadian, minat, dan kemampuan menggunakan algoritma **Decision Tree**.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-orange.svg)](https://scikit-learn.org/)

---

## ✨ Fitur Unggulan

| Fitur                  | Deskripsi |
|------------------------|-----------|
| 🧠 **AI Recommendation** | Sistem rekomendasi berbasis Decision Tree yang telah dilatih |
| 📊 **Analisis Multifaktor** | Menggabungkan minat, nilai akademik, dan kepribadian |
| 🎯 **Akurasi Tinggi**     | Model machine learning dengan akurasi >85% |
| 📱 **Web Responsif**     | Tampilan UI yang mendukung semua perangkat (mobile & desktop) |

---

## 🖼️ Screenshot Aplikasi

<div align="center">
  <img src="images/home.png" width="45%" alt="Halaman Utama">
  <img src="images/form.png" width="45%" alt="Form Input">
  <img src="images/hasil.png" width="45%" alt="Hasil Rekomendasi"> 
  <img src="images/chart.png" width="45%" alt="Chart Hasil"> 
  
</div>

---

## 🛠️ Arsitektur & Teknologi


```mermaid
graph TD
    A[Frontend] -->|Flask Template| B[Backend Flask]
    B --> C[Input User]
    C --> D[Decision Tree Model]
    D --> E[Dataset Processing]
    E --> F[Pandas/NumPy]
    D --> G[Output Rekomendasi]