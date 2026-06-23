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
MODEL_PATH = os.path.join(BASE_DIR, "model_postur.pkl")

# FUNGSI NORMALISASI MATEMATIS
def normalisasi_koordinat(baris_koordinat):
    coords = np.array(baris_koordinat).reshape(33, 3)
    tengah_bahu = (coords[11] + coords[12]) / 2.0
    tengah_pinggul = (coords[23] + coords[24]) / 2.0
    panjang_tulang = np.linalg.norm(tengah_bahu - tengah_pinggul)
    if panjang_tulang == 0: panjang_tulang = 1.0
    coords_normal = (coords - tengah_pinggul) / panjang_tulang
    return coords_normal.flatten().tolist()

print(f"1. Membaca dataset publik ({DATASET_PATH})...")
df_publik = pd.read_csv(DATASET_PATH)
pemetaan_label = {
    'TUP': 'Ergonomis', 'TLF': 'Non-Ergonomis', 'TLB': 'Non-Ergonomis',
    'TLL': 'Non-Ergonomis', 'TLR': 'Non-Ergonomis'
}
df_publik['upperbody_label'] = df_publik['upperbody_label'].replace(pemetaan_label)

y_publik = df_publik['upperbody_label'].values
X_publik = df_publik.drop(columns=['subject', 'upperbody_label', 'lowerbody_label']).values

if os.path.exists(DATA_RIIL_PATH):
    print(f"2. Membaca data riil buatan sendiri ({DATA_RIIL_PATH})...")
    df_riil = pd.read_csv(DATA_RIIL_PATH, header=None)
    y_riil = df_riil.iloc[:, 0].values
    X_riil = df_riil.iloc[:, 1:].values
    
    X_gabungan_mentah = np.vstack((X_publik, X_riil))
    y_gabungan = np.concatenate((y_publik, y_riil))
else:
    print(f"2. File {DATA_RIIL_PATH} tidak ditemukan, hanya menggunakan dataset publik.")
    X_gabungan_mentah = X_publik
    y_gabungan = y_publik

print("3. Menormalisasi skala vektor dari seluruh data...")
X_gabungan_normal = np.array([normalisasi_koordinat(baris) for baris in X_gabungan_mentah])

print("4. Memulai proses pembelajaran...")
X_train, X_test, y_train, y_test = train_test_split(X_gabungan_normal, y_gabungan, test_size=0.2, random_state=42)

# Tahap 1: Inisialisasi Decision Tree untuk seleksi fitur
dt_selector = DecisionTreeClassifier(max_depth=10, random_state=42)

# Tahap 2: Gunakan SelectFromModel untuk menyaring 15 fitur terbaik
selector = SelectFromModel(estimator=dt_selector, max_features=15, threshold=-np.inf)

# Tahap 3: Inisialisasi estimator Stacking (DT & KNN) yang akan memproses 15 fitur terpilih
dt_estimator = DecisionTreeClassifier(max_depth=10, random_state=42)
knn_estimator = KNeighborsClassifier(n_neighbors=5)

stacking_model = StackingClassifier(
    estimators=[('decision_tree', dt_estimator)],
    final_estimator=knn_estimator,
    passthrough=True
)

# Tahap 4: Gabungkan dalam Pipeline (Seleksi Fitur -> Stacking Classifier)
model_gabungan = Pipeline([
    ('feature_selection', selector),
    ('ensemble', stacking_model)
])

# Latih pipeline model
model_gabungan.fit(X_train, y_train)

prediksi = model_gabungan.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)
print(f"5. Latihan Selesai! Akurasi DT-KNN Pipeline (15 Fitur + Stacking): {akurasi * 100:.2f}%")

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_gabungan, f)
print(f"6. Otak AI (Decision Tree + KNN Stacking) berhasil disimpan di: {MODEL_PATH}")