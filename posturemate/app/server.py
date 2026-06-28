"""
PostureMate Backend API
This server provides endpoints to analyze body posture coordination using trained Machine Learning models.
"""

import os
import math
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PostureMate Backend API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def redirect_to_docs():
    """Redirect root access to Swagger API documentation."""
    return RedirectResponse(url="/docs")


# Load classification models on startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_BINER_PATH = os.path.join(BASE_DIR, "model_biner.pkl")
MODEL_MULTICLASS_PATH = os.path.join(BASE_DIR, "model_multiclass.pkl")

try:
    with open(MODEL_BINER_PATH, 'rb') as f_biner:
        model_biner = pickle.load(f_biner)
    with open(MODEL_MULTICLASS_PATH, 'rb') as f_multi:
        model_multi = pickle.load(f_multi)
except FileNotFoundError:
    raise RuntimeError("Pastikan file model_biner.pkl dan model_multiclass.pkl ada di folder!")


class InputPostur(BaseModel):
    koordinat: list[float]


def normalisasi_koordinat(baris_koordinat):
    """Normalize body landmark coordinates relative to hip center and spine length."""
    coords = np.array(baris_koordinat).reshape(33, 3)
    tengah_bahu = (coords[11] + coords[12]) / 2.0
    tengah_pinggul = (coords[23] + coords[24]) / 2.0
    panjang_tulang = np.linalg.norm(tengah_bahu - tengah_pinggul)
    if panjang_tulang == 0: panjang_tulang = 1.0
    coords_normal = (coords - tengah_pinggul) / panjang_tulang
    return coords_normal.flatten().tolist()


def hitung_metrik(baris_koordinat):
    """Calculate shoulder slope angle and neck-to-shoulder relative Z-distance."""
    coords = np.array(baris_koordinat).reshape(33, 3)
    dy = coords[12][1] - coords[11][1]
    dx = coords[12][0] - coords[11][0]
    kemiringan = abs(math.degrees(math.atan(dy / dx))) if dx != 0 else 90.0
    
    rata_z_bahu = (coords[11][2] + coords[12][2]) / 2
    rata_z_telinga = (coords[7][2] + coords[8][2]) / 2
    jarak_leher = (rata_z_bahu - rata_z_telinga) * 100
    return kemiringan, jarak_leher


@app.post("/api/deteksi")
def deteksi_postur_endpoint(data: InputPostur):
    """Analyze posture coordinates using binary and multiclass classification models."""
    if len(data.koordinat) != 99:
        raise HTTPException(status_code=400, detail="Harus 99 koordinat.")

    # Calculate metrics and normalize coordinates for model input
    kemiringan, jarak_leher = hitung_metrik(data.koordinat)
    koordinat_normal = normalisasi_koordinat(data.koordinat)
    X_input = np.array(koordinat_normal).reshape(1, -1)
    
    # Predict binary posture state and detailed multiclass labels
    tebakan_biner = model_biner.predict(X_input)[0]       # "Ergonomis" / "Non-Ergonomis"
    tebakan_detail = model_multi.predict(X_input)[0]      # e.g., "TUP", "TLF", "TLB"

    return {
        "prediksi_utama": tebakan_biner,
        "prediksi_detail": tebakan_detail,
        "pesan": "Berhasil diproses"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)