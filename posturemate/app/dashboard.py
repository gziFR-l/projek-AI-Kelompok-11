"""
PostureMate - Dashboard Analitik (Streamlit)

Menampilkan analitik dari riwayat sesi deteksi postur yang dicatat oleh
kamera_ai.py ke dalam SQLite (data/posturemate.db, tabel riwayat_postur),
serta eksplorasi dataset latih (data/dataset.csv).

Menjalankan:
    streamlit run posturemate/app/dashboard.py
"""

import os
import sqlite3

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Konfigurasi & Path
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "posturemate.db")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

# Pemetaan kode label dataset -> nama yang mudah dibaca
NAMA_LABEL = {
    "TUP": "Tegak (Ergonomis)",
    "TLF": "Condong ke Depan",
    "TLB": "Terlalu Bersandar",
    "TLL": "Miring Kiri",
    "TLR": "Miring Kanan",
}

st.set_page_config(
    page_title="PostureMate - Dashboard Analitik",
    page_icon="🪑",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Pemuatan Data
# ----------------------------------------------------------------------------
@st.cache_data(ttl=30)
def muat_riwayat() -> pd.DataFrame:
    """Membaca tabel riwayat_postur dari SQLite. Kembalikan DataFrame kosong
    bila DB / tabel belum ada (kamera_ai.py belum pernah dijalankan)."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT timestamp, status_postur, kemiringan_bahu, jarak_leher, pesan "
            "FROM riwayat_postur ORDER BY timestamp",
            conn,
        )
        conn.close()
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["tanggal"] = df["timestamp"].dt.date
        df["jam"] = df["timestamp"].dt.hour
    return df


@st.cache_data(ttl=300)
def muat_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATASET_PATH, usecols=["upperbody_label"])


# ----------------------------------------------------------------------------
# Header & Sidebar
# ----------------------------------------------------------------------------
st.title("🪑 PostureMate — Dashboard Analitik")
st.caption(
    "Visualisasi riwayat postur duduk dari sesi deteksi real-time. "
    "Data dicatat otomatis oleh `kamera_ai.py` setiap 5 detik."
)

df = muat_riwayat()

with st.sidebar:
    st.header("⚙️ Pengaturan")
    if st.button("🔄 Muat Ulang Data"):
        st.cache_data.clear()
        st.rerun()

    if not df.empty:
        tanggal_tersedia = sorted(df["tanggal"].unique())
        pilih_tanggal = st.multiselect(
            "Filter tanggal",
            options=tanggal_tersedia,
            default=tanggal_tersedia,
            format_func=lambda d: d.strftime("%d %b %Y"),
        )
        if pilih_tanggal:
            df = df[df["tanggal"].isin(pilih_tanggal)]

    st.divider()
    st.caption(f"📁 DB: `{os.path.relpath(DB_PATH, BASE_DIR)}`")


# ----------------------------------------------------------------------------
# Bila belum ada data riwayat
# ----------------------------------------------------------------------------
if df.empty:
    st.info(
        "**Belum ada data riwayat sesi.**\n\n"
        "Jalankan deteksi real-time terlebih dahulu agar dashboard memiliki data:\n\n"
        "```bash\npython posturemate/app/kamera_ai.py\n```\n\n"
        "Setelah beberapa sesi terekam, kembali ke halaman ini dan klik "
        "**🔄 Muat Ulang Data**."
    )
    # Tetap tampilkan eksplorasi dataset latih bila tersedia
    ds = muat_dataset()
    if not ds.empty:
        st.divider()
        st.subheader("🧬 Eksplorasi Dataset Latih")
        dist = ds["upperbody_label"].map(lambda x: NAMA_LABEL.get(x, x)).value_counts()
        st.bar_chart(dist)
        st.caption(f"Total {len(ds):,} sampel pada `dataset.csv`.")
    st.stop()


# ----------------------------------------------------------------------------
# Tab Analitik
# ----------------------------------------------------------------------------
tab_ringkasan, tab_tren, tab_masalah, tab_dataset = st.tabs(
    ["📊 Ringkasan Sesi", "📈 Tren Waktu", "🔍 Distribusi Masalah", "🧬 Dataset Latih"]
)

# Setiap baris ~ 5 detik (interval penyimpanan di kamera_ai.py)
DETIK_PER_BARIS = 5

with tab_ringkasan:
    total_catatan = len(df)
    jumlah_ergonomis = int((df["status_postur"] == "Ergonomis").sum())
    jumlah_non = total_catatan - jumlah_ergonomis
    persen_baik = (jumlah_ergonomis / total_catatan * 100) if total_catatan else 0
    total_menit = total_catatan * DETIK_PER_BARIS / 60

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Durasi Terpantau", f"{total_menit:.1f} mnt")
    c2.metric("Postur Baik (Ergonomis)", f"{persen_baik:.1f}%")
    c3.metric("Catatan Ergonomis", f"{jumlah_ergonomis:,}")
    c4.metric("Catatan Non-Ergonomis", f"{jumlah_non:,}")

    st.divider()
    col_kiri, col_kanan = st.columns([1, 1])
    with col_kiri:
        st.subheader("Proporsi Postur")
        st.bar_chart(df["status_postur"].value_counts())
    with col_kanan:
        st.subheader("Rata-rata Metrik")
        st.metric("Kemiringan Bahu (rata-rata)", f"{df['kemiringan_bahu'].mean():.1f}°")
        st.metric("Jarak Leher Z (rata-rata)", f"{df['jarak_leher'].mean():.1f} cm")

with tab_tren:
    st.subheader("Metrik Postur dari Waktu ke Waktu")
    df_idx = df.set_index("timestamp")[["kemiringan_bahu", "jarak_leher"]]
    st.line_chart(df_idx)
    st.caption(
        "Kemiringan bahu tinggi → bahu tidak rata. "
        "Jarak leher (Z) tinggi → kepala terlalu maju ke depan."
    )

    st.divider()
    st.subheader("Persentase Postur Baik per Hari")
    per_hari = (
        df.assign(baik=(df["status_postur"] == "Ergonomis").astype(int))
        .groupby("tanggal")["baik"]
        .mean()
        .mul(100)
    )
    st.bar_chart(per_hari)

    st.divider()
    st.subheader("Aktivitas per Jam")
    st.bar_chart(df["jam"].value_counts().sort_index())

with tab_masalah:
    st.subheader("Distribusi Jenis Peringatan Postur")
    df_non = df[df["status_postur"] == "Non-Ergonomis"].copy()
    if df_non.empty:
        st.success("Tidak ada catatan postur buruk pada rentang terpilih. 🎉")
    else:
        pesan_bersih = (
            df_non["pesan"].fillna("").str.strip().replace("", "(tanpa keterangan)")
        )
        dist_pesan = pesan_bersih.value_counts()
        st.bar_chart(dist_pesan)
        st.caption(
            f"Total {len(df_non):,} catatan non-ergonomis. "
            "Peringatan paling sering muncul di urutan teratas."
        )
        st.dataframe(
            dist_pesan.rename_axis("Pesan").reset_index(name="Jumlah"),
            use_container_width=True,
            hide_index=True,
        )

with tab_dataset:
    st.subheader("🧬 Distribusi Label Dataset Latih")
    ds = muat_dataset()
    if ds.empty:
        st.warning("`dataset.csv` tidak ditemukan.")
    else:
        dist = ds["upperbody_label"].map(lambda x: NAMA_LABEL.get(x, x)).value_counts()
        st.bar_chart(dist)
        st.caption(
            f"Total {len(ds):,} sampel. Dipakai untuk melatih model biner & "
            "multiclass di `src/latih_ai.py`."
        )
