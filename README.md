# Anggota Kelompok 11

| Nama | NIM | Peran dalam Proyek |
| :--- | :---: | :--- |
| Yusran Rizqi Laksono | L0124125 | Backend |
| Ghazi Fahmi Ramadhan | L0124130 | Frontend |
| Maria Dewi Handayani | L0124132 | UI/UX Design |

---

# PostureMate: Sistem Klasifikasi Postur Duduk Mahasiswa dengan Dashboard Analitik Berbasis MediaPipe

PostureMate adalah sistem klasifikasi postur duduk mahasiswa secara real-time yang dirancang untuk mendeteksi kebiasaan duduk tidak sehat dan memberikan visualisasi analitik melalui dashboard interaktif. Proyek ini dikembangkan menggunakan MediaPipe Pose untuk ekstraksi landmark tubuh, algoritma machine learning (KNN dan Decision Tree) untuk klasifikasi postur, dan Streamlit untuk dashboard analitik.

---

## Fitur Utama

- **Deteksi Postur Real-Time**: Deteksi postur duduk secara langsung menggunakan kamera/webcam.
- **Klasifikasi Postur**: Mengelompokkan postur tubuh menjadi beberapa kelas (contoh: *Sikap Tegak*, *Membungkuk/Slouching*, *Condong ke Depan*, *Condong ke Samping*).
- **Pengingat Cerdas**: Sistem peringatan jika pengguna berada dalam posisi salah/buruk terlalu lama.
- **Dashboard Analitik**: Analisis riwayat waktu duduk, persentase postur baik vs. buruk, serta grafik tren harian/mingguan.
- **Fitur Perekaman Dataset**: Modul untuk merekam data kustom langsung melalui webcam untuk melatih model baru.

---

## Arsitektur Sistem

Sistem PostureMate bekerja dengan alur kerja berikut:

1. **Input Video**: Mengambil frame video secara real-time dari kamera/webcam.
2. **Pose Landmark Extraction (MediaPipe)**: Mendeteksi 33 titik koordinat tubuh (*keypoints*), dengan fokus pada area bahu, leher, dan mata.
3. **Feature Engineering**: Menghitung jarak, sudut inklinasi leher, kelengkungan punggung, serta kemiringan bahu berdasarkan titik koordinat MediaPipe.
4. **Machine Learning Classification**: Menggunakan model *Machine Learning* berupa KNN dan Decision Tree untuk mengklasifikasikan pose tersebut.
5. **Real-time Feedback & Dashboard**: Menampilkan klasifikasi pose langsung di layar dan mencatat statistik ke dashboard Streamlit.
