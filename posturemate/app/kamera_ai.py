from typing import Any
import cv2
import mediapipe as mp
import pickle
import numpy as np
import warnings
import math
from collections import deque

import os

warnings.filterwarnings("ignore", category=UserWarning)

# FUNGSI NORMALISASI MATEMATIS
def normalisasi_koordinat(baris_koordinat):
    coords = np.array(baris_koordinat).reshape(33, 3)
    tengah_bahu = (coords[11] + coords[12]) / 2.0
    tengah_pinggul = (coords[23] + coords[24]) / 2.0
    panjang_tulang = np.linalg.norm(tengah_bahu - tengah_pinggul)
    if panjang_tulang == 0: panjang_tulang = 1.0
    coords_normal = (coords - tengah_pinggul) / panjang_tulang
    return coords_normal.flatten().tolist()

# Tentukan path absolut untuk model_postur.pkl
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_postur.pkl")

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("BERHASIL: Otak AI (model_postur.pkl) dimuat!")
except FileNotFoundError:
    print(f"GAGAL: File model_postur.pkl tidak ditemukan di {MODEL_PATH}. Jalankan latih_ai.py terlebih dahulu!")
    exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

history_kemiringan = deque(maxlen=10) 
history_leher_z = deque(maxlen=10)
status_terakhir = "Ergonomis" 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = None
    # Prioritaskan indeks kamera eksternal (biasanya 1 atau 2) sebelum fallback ke kamera bawaan (indeks 0)
    for idx in [1, 2, 0]:
        temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(idx)
        
        if temp_cap.isOpened():
            # Uji coba membaca beberapa frame untuk memastikan stream benar-benar aktif
            success = True
            for _ in range(5):
                ret, _ = temp_cap.read()
                if not ret:
                    success = False
                    break
            if success:
                cap = temp_cap
                print(f"Kamera berhasil dibuka menggunakan indeks {idx}.")
                break
            else:
                temp_cap.release()

    if cap is None:
        print("ERROR: Kamera/Webcam tidak dapat dibuka atau tidak dapat membaca frame.")
        print("Pastikan:")
        print("1. Kamera fisik terhubung ke komputer.")
        print("2. Kamera tidak sedang digunakan oleh aplikasi lain (seperti Zoom, Teams, browser, dll.).")
        print("3. Driver kamera Anda sudah terpasang dengan benar.")
        exit()

    print("Memulai deteksi postur...")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Peringatan: Gagal membaca frame dari kamera. Menghentikan program...")
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results: Any = pose.process(image_rgb)

        if results.pose_landmarks and results.pose_world_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, list(mp_pose.POSE_CONNECTIONS),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )

            landmarks = results.pose_landmarks.landmark
            world_landmarks = results.pose_world_landmarks.landmark

            bahu_kiri = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            bahu_kanan = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            dy = bahu_kanan.y - bahu_kiri.y
            dx = bahu_kanan.x - bahu_kiri.x
            sudut_mentah = abs(math.degrees(math.atan(dy / dx))) if dx != 0 else 90.0

            history_kemiringan.append(sudut_mentah)
            kemiringan_halus = sum(history_kemiringan) / len(history_kemiringan)

            w_bahu_kiri = world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            w_bahu_kanan = world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            w_telinga_kiri = world_landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            w_telinga_kanan = world_landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

            rata_z_bahu = (w_bahu_kiri.z + w_bahu_kanan.z) / 2
            rata_z_telinga = (w_telinga_kiri.z + w_telinga_kanan.z) / 2
            jarak_z_mentah = (rata_z_bahu - rata_z_telinga) * 100
            
            history_leher_z.append(jarak_z_mentah)
            jarak_leher_halus = sum(history_leher_z) / len(history_leher_z)

            h, w, _ = image.shape
            b_kiri_px = (int(bahu_kiri.x * w), int(bahu_kiri.y * h))
            b_kanan_px = (int(bahu_kanan.x * w), int(bahu_kanan.y * h))
            p_kiri_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w), 
                         int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h))
            p_kanan_px = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w), 
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h))
            t_kiri_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w), 
                         int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h))
            t_kanan_px = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w), 
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h))

            tengah_bahu = ((b_kiri_px[0] + b_kanan_px[0]) // 2, (b_kiri_px[1] + b_kanan_px[1]) // 2)
            tengah_pinggul = ((p_kiri_px[0] + p_kanan_px[0]) // 2, (p_kiri_px[1] + p_kanan_px[1]) // 2)
            tengah_telinga = ((t_kiri_px[0] + t_kanan_px[0]) // 2, (t_kiri_px[1] + t_kanan_px[1]) // 2)

            cv2.line(image, tengah_bahu, tengah_pinggul, (255, 255, 0), 3) 
            cv2.line(image, tengah_bahu, tengah_telinga, (0, 165, 255), 3)

            # PREDIKSI AI DENGAN DATA NORMALISASI
            koordinat_tubuh = []
            for landmark in world_landmarks:
                koordinat_tubuh.extend([landmark.x, landmark.y, landmark.z])
            
            koordinat_normal = normalisasi_koordinat(koordinat_tubuh)
            X_input = np.array(koordinat_normal).reshape(1, -1)
            tebakan_ai = model.predict(X_input)[0]

            pesan_tambahan = ""

            # LOGIKA HYSTERESIS & PESAN KUSTOM
            if status_terakhir == "Ergonomis":
                if kemiringan_halus > 6.0: 
                    status_terakhir = "Non-Ergonomis"
                    pesan_tambahan = "(Perbaiki Bahumu)"
                elif jarak_leher_halus > 9.0: 
                    status_terakhir = "Non-Ergonomis"
                    pesan_tambahan = "(Tarik Lehermu)"
                elif tebakan_ai == "Non-Ergonomis":
                    status_terakhir = "Non-Ergonomis"
                    pesan_tambahan = "(Tegakkan Punggung)"
            else:
                # Syarat untuk kembali hijau DILONGGARKAN
                if kemiringan_halus < 2.5 and jarak_leher_halus < 8.0 and tebakan_ai == "Ergonomis":
                    status_terakhir = "Ergonomis"
                    pesan_tambahan = ""
                else:
                    # Menjaga pesan tetap relevan dengan batas yang lebih longgar
                    if kemiringan_halus >= 3.5:
                        pesan_tambahan = "(Perbaiki Bahumu)"
                    elif jarak_leher_halus >= 8.5: 
                        pesan_tambahan = "(Tarik Lehermu)"
                    else:
                        pesan_tambahan = "(Tegakkan Punggung)"

            tebakan_final = f"{status_terakhir} {pesan_tambahan}".strip()
            warna_teks = (0, 0, 255) if status_terakhir == 'Non-Ergonomis' else (0, 255, 0)

            # Menampilkan UI
            cv2.rectangle(image, (0, 0), (650, 100), (0, 0, 0), -1) 
            cv2.putText(image, f"Status: {tebakan_final}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna_teks, 2, cv2.LINE_AA)
            cv2.putText(image, f"Kemiringan Bahu: {kemiringan_halus:.1f} Derajat", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"Jarak Leher (Z): {jarak_leher_halus:.1f} cm", (10, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Deteksi Postur Cerdas - PostureMate', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()