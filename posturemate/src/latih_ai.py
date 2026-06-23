import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# Menentukan path absolut berdasarkan lokasi file latih_ai.py ini
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
DATA_RIIL_PATH = os.path.join(BASE_DIR, "data", "data_riil.csv")

MODEL_BINER_PATH = os.path.join(BASE_DIR, "model_biner.pkl")
MODEL_MULTICLASS_PATH = os.path.join(BASE_DIR, "model_multiclass.pkl")

# FUNGSI NORMALISASI MATEMATIS
def normalisasi_koordinat(baris_koordinat):
    coords = np.array(baris_koordinat).reshape(33, 3)
    tengah_bahu = (coords[11] + coords[12]) / 2.0
    tengah_pinggul = (coords[23] + coords[24]) / 2.0
    panjang_tulang = np.linalg.norm(tengah_bahu - tengah_pinggul)
    if panjang_tulang == 0: panjang_tulang = 1.0
    coords_normal = (coords - tengah_pinggul) / panjang_tulang
    return coords_normal.flatten().tolist()

# ==========================================
# TAHAP 1: MELATIH MODEL BINER (TINGKAT 1)
# ==========================================
print("=== TRAINING MODEL BINER (TINGKAT 1) ===")
print(f"1. Membaca dataset publik ({DATASET_PATH})...")
df_publik_raw = pd.read_csv(DATASET_PATH)

# Ubah label publik menjadi biner: Ergonomis vs Non-Ergonomis
pemetaan_label = {
    'TUP': 'Ergonomis', 'TLF': 'Non-Ergonomis', 'TLB': 'Non-Ergonomis',
    'TLL': 'Non-Ergonomis', 'TLR': 'Non-Ergonomis'
}
y_publik_biner = np.array(df_publik_raw['upperbody_label'].replace(pemetaan_label).tolist())
X_publik_biner = df_publik_raw.drop(columns=['subject', 'upperbody_label', 'lowerbody_label']).values

if os.path.exists(DATA_RIIL_PATH):
    print(f"2. Membaca data riil buatan sendiri ({DATA_RIIL_PATH})...")
    df_riil = pd.read_csv(DATA_RIIL_PATH, header=None)
    y_riil = df_riil.iloc[:, 0].values
    X_riil = df_riil.iloc[:, 1:].values
    
    X_gabungan_mentah = np.vstack((X_publik_biner, X_riil))
    y_gabungan = np.concatenate((y_publik_biner, y_riil))
else:
    print(f"2. File {DATA_RIIL_PATH} tidak ditemukan, hanya menggunakan dataset publik.")
    X_gabungan_mentah = X_publik_biner
    y_gabungan = y_publik_biner

print("3. Menormalisasi data...")
X_gabungan_normal = np.array([normalisasi_koordinat(baris) for baris in X_gabungan_mentah])
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_gabungan_normal, y_gabungan, test_size=0.2, random_state=42
)

# Pipeline Biner: Seleksi Fitur -> Stacking (DT + KNN)
dt_selector_bin = DecisionTreeClassifier(max_depth=10, random_state=42)
selector_bin = SelectFromModel(estimator=dt_selector_bin, max_features=15, threshold=-np.inf)

dt_est_bin = DecisionTreeClassifier(max_depth=10, random_state=42)
knn_est_bin = KNeighborsClassifier(n_neighbors=5)
stacking_bin = StackingClassifier(
    estimators=[('decision_tree', dt_est_bin)],
    final_estimator=knn_est_bin,
    passthrough=True
)

model_biner = Pipeline([
    ('feature_selection', selector_bin),
    ('ensemble', stacking_bin)
])

print("4. Melatih model biner...")
model_biner.fit(X_train_bin, y_train_bin)

prediksi_bin = model_biner.predict(X_test_bin)
akurasi_bin = accuracy_score(y_test_bin, prediksi_bin)
print(f"5. Selesai! Akurasi Model Biner: {akurasi_bin * 100:.2f}%")

with open(MODEL_BINER_PATH, 'wb') as f:
    pickle.dump(model_biner, f)
print(f"6. Model Biner berhasil disimpan di: {MODEL_BINER_PATH}\n")


# ==========================================
# TAHAP 2: MELATIH MODEL MULTICLASS (TINGKAT 2)
# ==========================================
print("=== TRAINING MODEL MULTICLASS (TINGKAT 2) ===")
print("1. Menyaring dataset publik untuk menyisakan posisi Non-Ergonomis saja (TLF, TLB, TLL, TLR)...")

# Saring baris yang BUKAN 'TUP'
df_multiclass_raw = df_publik_raw[df_publik_raw['upperbody_label'] != 'TUP'].copy()
y_publik_multi = np.array(df_multiclass_raw['upperbody_label'].tolist())
X_publik_multi = df_multiclass_raw.drop(columns=['subject', 'upperbody_label', 'lowerbody_label']).values

print(f"   Jumlah sampel non-ergonomis terkumpul: {len(df_multiclass_raw)}")
print("2. Menormalisasi data...")
X_multi_normal = np.array([normalisasi_koordinat(baris) for baris in X_publik_multi])
X_train_mul, X_test_mul, y_train_mul, y_test_mul = train_test_split(
    X_multi_normal, y_publik_multi, test_size=0.2, random_state=42
)

# Pipeline Multiclass: Seleksi Fitur -> Stacking (DT + KNN)
dt_selector_mul = DecisionTreeClassifier(max_depth=10, random_state=42)
selector_mul = SelectFromModel(estimator=dt_selector_mul, max_features=15, threshold=-np.inf)

dt_est_mul = DecisionTreeClassifier(max_depth=10, random_state=42)
knn_est_mul = KNeighborsClassifier(n_neighbors=5)
stacking_mul = StackingClassifier(
    estimators=[('decision_tree', dt_est_mul)],
    final_estimator=knn_est_mul,
    passthrough=True
)

model_multiclass = Pipeline([
    ('feature_selection', selector_mul),
    ('ensemble', stacking_mul)
])

print("3. Melatih model multiclass...")
model_multiclass.fit(X_train_mul, y_train_mul)

prediksi_mul = model_multiclass.predict(X_test_mul)
akurasi_mul = accuracy_score(y_test_mul, prediksi_mul)
print(f"4. Selesai! Akurasi Model Multiclass: {akurasi_mul * 100:.2f}%")

with open(MODEL_MULTICLASS_PATH, 'wb') as f:
    pickle.dump(model_multiclass, f)
print(f"5. Model Multiclass berhasil disimpan di: {MODEL_MULTICLASS_PATH}\n")