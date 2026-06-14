# 👥 Anggota Kelompok 11

| Nama | NIM | Peran dalam Proyek |
| :--- | :---: | :--- |
| Yusran Rizqi Laksono | L0124125 | Backend |
| Ghazi Fahmi Ramadhan | L0124130 | Frontend |
| Maria Dewi Handayani | L0124132 | ML Engineer |

---

# PostureMate: Sistem Klasifikasi Postur Duduk Mahasiswa dengan Dashboard Analitik Berbasis MediaPipe

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0%2B-green.svg)](https://google.github.io/mediapipe/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20.0%2B-red.svg)](https://streamlit.io/)

**PostureMate** adalah sistem klasifikasi postur duduk mahasiswa secara real-time yang dirancang untuk mendeteksi kebiasaan duduk tidak sehat dan memberikan visualisasi analitik melalui dashboard interaktif. Proyek ini dikembangkan menggunakan **MediaPipe Pose** untuk ekstraksi landmark tubuh, algoritma machine learning (**KNN** & **Decision Tree**) untuk klasifikasi postur, dan **Streamlit** untuk dashboard analitik.

---

## 🌟 Fitur Utama

- **Deteksi Postur Real-Time**: Deteksi postur duduk secara langsung menggunakan kamera/webcam.
- **Klasifikasi Postur**: Mengelompokkan postur tubuh menjadi beberapa kelas (contoh: *Sikap Tegak*, *Membungkuk/Slouching*, *Condong ke Depan*, *Condong ke Samping*).
- **Pengingat Cerdas**: Sistem peringatan jika pengguna berada dalam posisi salah/buruk terlalu lama.
- **Dashboard Analitik**: Analisis riwayat waktu duduk, persentase postur baik vs. buruk, serta grafik tren harian/mingguan.
- **Fitur Perekaman Dataset**: Modul untuk merekam data kustom langsung melalui webcam untuk melatih model baru.

---

## 🏗️ Arsitektur Sistem

Sistem PostureMate bekerja dengan alur kerja berikut:

1. **Input Video**: Mengambil frame video secara real-time dari kamera/webcam.
2. **Pose Landmark Extraction (MediaPipe)**: Mendeteksi 33 titik koordinat tubuh (*keypoints*), dengan fokus pada area bahu, leher, telinga, mata, dan pinggul.
3. **Feature Engineering**: Menghitung jarak, sudut inklinasi leher, kelengkungan punggung, serta kemiringan bahu berdasarkan titik koordinat MediaPipe.
4. **Machine Learning Classification**: Menggunakan model ML (seperti KNN atau Decision Tree) untuk mengklasifikasikan pose tersebut.
5. **Real-time Feedback & Dashboard**: Menampilkan klasifikasi pose langsung di layar dan mencatat statistik ke dashboard Streamlit.

---

## 🛠️ Instalasi & Setup

### Prasyarat
Pastikan Anda sudah menginstal **Python 3.11** di komputer Anda.

### Langkah-langkah
1. **Clone Repositori**
   ```bash
   git clone https://github.com/username/repo-name.git
   cd repo-name
   ```

2. **Buat dan Aktifkan Virtual Environment (Opsional tetapi Direkomendasikan)**
   * Di Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   * Di macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Instal Dependencies**
   ```bash
   pip install -r posturemate/requirements.txt
   ```

4. **Uji Coba MediaPipe**
   Pastikan webcam dan MediaPipe bekerja dengan baik:
   ```bash
   python posturemate/src/test_mediapipe.py
   ```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Perekaman Dataset Kustom (Jika Ingin Membuat Data Baru)
Gunakan modul perekam untuk mengumpulkan koordinat tubuh dengan posisi duduk yang berbeda:
```bash
python posturemate/src/rekam_dataset.py
```

### 2. Pelatihan Model
Setelah mengumpulkan dataset di folder `data/`, jalankan modul training untuk melatih model klasifikasi KNN & Decision Tree:
```bash
python posturemate/src/train_model.py
```

### 3. Jalankan Deteksi Postur Real-time
Jalankan aplikasi utama untuk memulai monitoring postur duduk secara langsung:
```bash
python posturemate/app/main.py
```

### 4. Jalankan Dashboard Analitik
Jalankan dashboard Streamlit untuk melihat laporan, statistik, dan tren perilaku duduk Anda:
```bash
streamlit run posturemate/app/dashboard.py
```

---

## 📊 Dataset & Fitur yang Digunakan
Fitur input klasifikasi diekstrak dari landmark MediaPipe Pose (3D Coordinates):
* **Sudut Leher (Neck Angle)**: Sudut antara bahu dan telinga untuk mendeteksi *forward head posture*.
* **Kelengkungan Punggung (Torso Angle)**: Sudut inklinasi punggung terhadap garis vertikal bumi.
* **Kemiringan Bahu (Shoulder Level)**: Perbedaan tinggi antara bahu kiri dan kanan untuk mendeteksi posisi duduk miring.
* **Jarak ke Kamera (Distance)**: Jarak relatif wajah ke kamera untuk memantau jika mahasiswa duduk terlalu dekat dengan layar.
