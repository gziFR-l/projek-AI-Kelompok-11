import os
import math
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# 1. Inisialisasi Aplikasi API
app = FastAPI(title="PostureMate Backend API", version="1.0")

# 2. Tambahkan Middleware CORS (mengizinkan koneksi dari frontend/dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Halaman Utama (Langsung pindah ke Swagger UI)
@app.get("/")
def halaman_utama():
    return RedirectResponse(url="/docs")

# 3. Muat Otak AI secara dinamis dengan path absolut (Model Biner & Multiclass)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_BINER_PATH = os.path.join(BASE_DIR, "model_biner.pkl")
MODEL_MULTICLASS_PATH = os.path.join(BASE_DIR, "model_multiclass.pkl")

model_biner = None
model_multiclass = None

try:
    with open(MODEL_BINER_PATH, 'rb') as f:
        model_biner = pickle.load(f)
    with open(MODEL_MULTICLASS_PATH, 'rb') as f:
        model_multiclass = pickle.load(f)
    print("Kedua model AI berhasil dimuat.")
except FileNotFoundError:
    # Mengamankan agar server tetap bisa berjalan/start meskipun model belum dilatih
    print(f"Peringatan: File model_biner.pkl atau model_multiclass.pkl belum lengkap!")

# 4. Kontrak Data
class InputPostur(BaseModel):
    koordinat: list[float]

# --- FUNGSI MATEMATIS ---
def normalisasi_koordinat(baris_koordinat):
    coords = np.array(baris_koordinat).reshape(33, 3)
    tengah_bahu = (coords[11] + coords[12]) / 2.0
    tengah_pinggul = (coords[23] + coords[24]) / 2.0
    panjang_tulang = np.linalg.norm(tengah_bahu - tengah_pinggul)
    if panjang_tulang == 0: panjang_tulang = 1.0
    coords_normal = (coords - tengah_pinggul) / panjang_tulang
    return coords_normal.flatten().tolist()

def hitung_metrik(baris_koordinat):
    coords = np.array(baris_koordinat).reshape(33, 3)
    dy = coords[12][1] - coords[11][1]
    dx = coords[12][0] - coords[11][0]
    kemiringan = abs(math.degrees(math.atan(dy / dx))) if dx != 0 else 90.0
    
    rata_z_bahu = (coords[11][2] + coords[12][2]) / 2
    rata_z_telinga = (coords[7][2] + coords[8][2]) / 2
    jarak_leher = (rata_z_bahu - rata_z_telinga) * 100
    return kemiringan, jarak_leher

# 5. Endpoint Deteksi (Model Bertingkat)
@app.post("/api/deteksi")
def deteksi_postur_endpoint(data: InputPostur):
    if model_biner is None or model_multiclass is None:
        raise HTTPException(
            status_code=503, 
            detail="Model AI (model_biner.pkl & model_multiclass.pkl) belum siap. Silakan jalankan latih_ai.py terlebih dahulu."
        )
        
    if len(data.koordinat) != 99:
        raise HTTPException(status_code=400, detail="Data harus berisi tepat 99 angka koordinat.")

    kemiringan, jarak_leher = hitung_metrik(data.koordinat)
    koordinat_normal = normalisasi_koordinat(data.koordinat)
    X_input = np.array(koordinat_normal).reshape(1, -1)
    
    # Tingkat 1: Prediksi Biner
    tebakan_ai = model_biner.predict(X_input)[0]
    
    # Tingkat 2: Prediksi Detail Multiclass jika dideteksi Non-Ergonomis
    tebakan_detail = ""
    pesan_rekomendasi = "Aman"
    if tebakan_ai == "Non-Ergonomis":
        tebakan_detail = model_multiclass.predict(X_input)[0]
        
        # Pemetaan pesan rekomendasi
        if tebakan_detail == "TLF":
            pesan_rekomendasi = "Tegakkan Punggung"
        elif tebakan_detail == "TLB":
            pesan_rekomendasi = "Badan Terlalu Bersandar"
        elif tebakan_detail == "TLL":
            pesan_rekomendasi = "Miring Kiri"
        elif tebakan_detail == "TLR":
            pesan_rekomendasi = "Miring Kanan"
        else:
            pesan_rekomendasi = "Tegakkan Punggung"

    return {
        "prediksi_ai": tebakan_ai,
        "detail_non_ergonomis": tebakan_detail,
        "metrik": {
            "kemiringan_bahu_derajat": round(kemiringan, 2),
            "jarak_leher_cm": round(jarak_leher, 2)
        },
        "pesan": pesan_rekomendasi
    }

# 6. Runner Block untuk menjalankan python server.py secara langsung
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)