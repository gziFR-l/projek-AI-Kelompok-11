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

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ----------------------------------------------------------------------------
# Konfigurasi & Path
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "posturemate.db")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

# Palet warna brand
INDIGO = "#6366F1"
UNGU = "#8B5CF6"
HIJAU = "#10B981"
MERAH = "#EF4444"
KUNING = "#F59E0B"

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
# Gaya / CSS Kustom
# ----------------------------------------------------------------------------
def suntik_css() -> None:
    st.markdown(
        """
        <style>
        /* Sembunyikan elemen bawaan agar lebih bersih */
        #MainMenu, footer {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}

        /* Header hero dengan gradien */
        .hero {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 60%, #A855F7 100%);
            padding: 1.6rem 2rem;
            border-radius: 20px;
            color: #FFFFFF;
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
            margin-bottom: 1.4rem;
        }
        .hero h1 {margin: 0; font-size: 2rem; font-weight: 800; color: #FFFFFF;}
        .hero p {margin: 0.35rem 0 0; opacity: 0.92; font-size: 0.98rem;}

        /* Banner status live */
        .status-banner {
            display: flex; align-items: center; gap: 14px;
            padding: 1.1rem 1.5rem; border-radius: 16px;
            margin-bottom: 1.2rem; font-weight: 700;
            border: 1px solid rgba(0,0,0,0.05);
        }
        .status-dot {width: 14px; height: 14px; border-radius: 50%;
            box-shadow: 0 0 0 6px rgba(255,255,255,0.45);}
        .status-baik {background: #ECFDF5; color: #065F46;}
        .status-buruk {background: #FEF2F2; color: #991B1B;}

        /* Kartu metrik */
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
            transition: transform .15s ease, box-shadow .15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
        }
        div[data-testid="stMetricLabel"] p {color: #64748B; font-weight: 600;}
        div[data-testid="stMetricValue"] {color: #0F172A; font-weight: 800;}

        /* Tab */
        button[data-baseweb="tab"] {font-weight: 600;}

        /* Sub-judul section */
        h3 {color: #1E293B; font-weight: 700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def gaya_grafik(c: alt.Chart) -> alt.Chart:
    """Terapkan gaya konsisten yang bersih pada chart Altair."""
    return (
        c.configure_view(strokeWidth=0)
        .configure_axis(
            grid=True,
            gridColor="#EEF2F7",
            domain=False,
            labelColor="#64748B",
            titleColor="#475569",
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_legend(labelColor="#475569", titleColor="#334155")
    )


# ----------------------------------------------------------------------------
# Pemuatan Data
# ----------------------------------------------------------------------------
@st.cache_data(ttl=5)
def muat_riwayat() -> pd.DataFrame:
    """Membaca tabel riwayat_postur dari SQLite. Kembalikan DataFrame kosong
    bila DB / tabel belum ada (kamera_ai.py belum pernah dijalankan)."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql_query(
                "SELECT timestamp, status_postur, kemiringan_bahu, jarak_leher, pesan, detail_postur "
                "FROM riwayat_postur ORDER BY timestamp",
                conn,
            )
        except (sqlite3.Error, pd.errors.DatabaseError):
            df = pd.read_sql_query(
                "SELECT timestamp, status_postur, kemiringan_bahu, jarak_leher, pesan "
                "FROM riwayat_postur ORDER BY timestamp",
                conn,
            )
            df["detail_postur"] = ""
        conn.close()
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["tanggal"] = df["timestamp"].dt.date  # type: ignore
        df["jam"] = df["timestamp"].dt.hour  # type: ignore
    return df


@st.cache_data(ttl=300)
def muat_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATASET_PATH, usecols=["upperbody_label"])


# ----------------------------------------------------------------------------
# Header & Sidebar
# ----------------------------------------------------------------------------
suntik_css()

st.markdown(
    """
    <div class="hero">
        <h1>🪑 PostureMate — Dashboard Analitik</h1>
        <p>Visualisasi riwayat postur duduk dari sesi deteksi real-time.
        Data dicatat otomatis oleh <code style="color:#FDE68A">kamera_ai.py</code> setiap 5 detik.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Pengaturan")

    auto_refresh = st.toggle("🔄 Auto-refresh (5 detik)", value=True)
    if auto_refresh:
        # Rerun otomatis tiap 5 detik agar data baru dari kamera ikut tampil
        st_autorefresh(interval=5000, key="auto_refresh_dashboard")
        st.caption("Dashboard memuat ulang data otomatis tiap 5 detik.")

    if st.button("🔄 Muat Ulang Data Sekarang", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

df = muat_riwayat()

with st.sidebar:
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
        "**🔄 Muat Ulang Data Sekarang**."
    )
    # Tetap tampilkan eksplorasi dataset latih bila tersedia
    ds = muat_dataset()
    if not ds.empty:
        st.divider()
        st.subheader("🧬 Eksplorasi Dataset Latih")
        dist = (
            ds["upperbody_label"].map(lambda x: NAMA_LABEL.get(x, x)).value_counts()
            .rename_axis("Kategori").reset_index(name="Jumlah")
        )
        chart = (
            alt.Chart(dist)
            .mark_bar(cornerRadiusEnd=5, color=INDIGO)
            .encode(
                x=alt.X("Jumlah:Q", title="Jumlah Sampel"),
                y=alt.Y("Kategori:N", sort="-x", title=None),
                tooltip=["Kategori", "Jumlah"],
            )
            .properties(height=240)
        )
        st.altair_chart(gaya_grafik(chart), use_container_width=True)
        st.caption(f"Total {len(ds):,} sampel pada `dataset.csv`.")
    st.stop()


# ----------------------------------------------------------------------------
# Banner Status Terkini
# ----------------------------------------------------------------------------
status_terkini = df.iloc[-1]["status_postur"]
waktu_terkini = df.iloc[-1]["timestamp"].strftime("%d %b %Y, %H:%M:%S")
if status_terkini == "Ergonomis":
    st.markdown(
        f"""<div class="status-banner status-baik">
        <span class="status-dot" style="background:{HIJAU}"></span>
        Postur terakhir: ERGONOMIS ✅
        <span style="font-weight:500; opacity:.75; margin-left:auto">Diperbarui {waktu_terkini}</span>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""<div class="status-banner status-buruk">
        <span class="status-dot" style="background:{MERAH}"></span>
        Postur terakhir: NON-ERGONOMIS ⚠️
        <span style="font-weight:500; opacity:.75; margin-left:auto">Diperbarui {waktu_terkini}</span>
        </div>""",
        unsafe_allow_html=True,
    )


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
        src = (
            df["status_postur"].value_counts()
            .rename_axis("Status").reset_index(name="Jumlah")
        )
        donut = (
            alt.Chart(src)
            .mark_arc(innerRadius=70, cornerRadius=4)
            .encode(
                theta=alt.Theta("Jumlah:Q", stack=True),
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(
                        domain=["Ergonomis", "Non-Ergonomis"], range=[HIJAU, MERAH]
                    ),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["Status", "Jumlah"],
            )
            .properties(height=280)
        )
        st.altair_chart(gaya_grafik(donut), use_container_width=True)
    with col_kanan:
        st.subheader("Rata-rata Metrik")
        st.metric("Kemiringan Bahu (rata-rata)", f"{df['kemiringan_bahu'].mean():.1f}°")
        st.metric("Jarak Leher Z (rata-rata)", f"{df['jarak_leher'].mean():.1f} cm")

with tab_tren:
    st.subheader("Metrik Postur dari Waktu ke Waktu")
    df_long = (
        df[["timestamp", "kemiringan_bahu", "jarak_leher"]]
        .rename(columns={"kemiringan_bahu": "Kemiringan Bahu", "jarak_leher": "Jarak Leher (Z)"})
        .melt("timestamp", var_name="Metrik", value_name="Nilai")
    )
    garis = (
        alt.Chart(df_long)
        .mark_line(strokeWidth=2, point=False)
        .encode(
            x=alt.X("timestamp:T", title="Waktu"),
            y=alt.Y("Nilai:Q", title=None),
            color=alt.Color(
                "Metrik:N",
                scale=alt.Scale(range=[INDIGO, KUNING]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[alt.Tooltip("timestamp:T", title="Waktu"), "Metrik", alt.Tooltip("Nilai:Q", format=".1f")],
        )
        .properties(height=300)
    )
    st.altair_chart(gaya_grafik(garis), use_container_width=True)
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
        .rename_axis("Tanggal").reset_index(name="Persen")
    )
    per_hari["Tanggal"] = per_hari["Tanggal"].astype(str)
    bar_hari = (
        alt.Chart(per_hari)
        .mark_bar(cornerRadiusEnd=5, color=HIJAU)
        .encode(
            x=alt.X("Tanggal:N", title=None),
            y=alt.Y("Persen:Q", title="% Ergonomis", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Tanggal", alt.Tooltip("Persen:Q", format=".1f")],
        )
        .properties(height=260)
    )
    st.altair_chart(gaya_grafik(bar_hari), use_container_width=True)

    st.divider()
    st.subheader("Aktivitas per Jam")
    per_jam = (
        df["jam"].value_counts().sort_index()
        .rename_axis("Jam").reset_index(name="Jumlah")
    )
    bar_jam = (
        alt.Chart(per_jam)
        .mark_bar(cornerRadiusEnd=5, color=INDIGO)
        .encode(
            x=alt.X("Jam:O", title="Jam (24h)"),
            y=alt.Y("Jumlah:Q", title="Catatan"),
            tooltip=["Jam", "Jumlah"],
        )
        .properties(height=240)
    )
    st.altair_chart(gaya_grafik(bar_jam), use_container_width=True)

with tab_masalah:
    st.subheader("Distribusi Jenis Peringatan Postur")
    df_non = df[df["status_postur"] == "Non-Ergonomis"].copy()
    if df_non.empty:
        st.success("Tidak ada catatan postur buruk pada rentang terpilih. 🎉")
    else:
        pesan_bersih = (
            df_non["pesan"].fillna("").str.strip().replace("", "(tanpa keterangan)")
        )
        dist_pesan = (
            pesan_bersih.value_counts().rename_axis("Pesan").reset_index(name="Jumlah")
        )
        bar_pesan = (
            alt.Chart(dist_pesan)
            .mark_bar(cornerRadiusEnd=5, color=MERAH)
            .encode(
                x=alt.X("Jumlah:Q", title="Jumlah"),
                y=alt.Y("Pesan:N", sort="-x", title=None),
                tooltip=["Pesan", "Jumlah"],
            )
            .properties(height=max(180, 42 * len(dist_pesan)))
        )
        st.altair_chart(gaya_grafik(bar_pesan), use_container_width=True)
        st.caption(
            f"Total {len(df_non):,} catatan non-ergonomis. "
            "Peringatan paling sering muncul di urutan teratas."
        )

        st.divider()
        st.subheader("Distribusi Klasifikasi Detail Postur (Model Multiclass)")
        detail_bersih = df_non["detail_postur"].fillna("").str.strip()
        detail_valid = detail_bersih[detail_bersih != ""]
        if detail_valid.empty:
            st.info("Belum ada data detail klasifikasi postur untuk rentang ini.")
        else:
            dist_detail = (
                detail_valid.map(lambda x: NAMA_LABEL.get(x, x))
                .value_counts().rename_axis("Klasifikasi").reset_index(name="Jumlah")
            )
            bar_detail = (
                alt.Chart(dist_detail)
                .mark_bar(cornerRadiusEnd=5, color=KUNING)
                .encode(
                    x=alt.X("Jumlah:Q", title="Jumlah"),
                    y=alt.Y("Klasifikasi:N", sort="-x", title=None),
                    tooltip=["Klasifikasi", "Jumlah"],
                )
                .properties(height=max(180, 42 * len(dist_detail)))
            )
            st.altair_chart(gaya_grafik(bar_detail), use_container_width=True)

with tab_dataset:
    st.subheader("🧬 Distribusi Label Dataset Latih")
    ds = muat_dataset()
    if ds.empty:
        st.warning("`dataset.csv` tidak ditemukan.")
    else:
        dist = (
            ds["upperbody_label"].map(lambda x: NAMA_LABEL.get(x, x)).value_counts()
            .rename_axis("Kategori").reset_index(name="Jumlah")
        )
        bar_ds = (
            alt.Chart(dist)
            .mark_bar(cornerRadiusEnd=5, color=UNGU)
            .encode(
                x=alt.X("Jumlah:Q", title="Jumlah Sampel"),
                y=alt.Y("Kategori:N", sort="-x", title=None),
                tooltip=["Kategori", "Jumlah"],
            )
            .properties(height=260)
        )
        st.altair_chart(gaya_grafik(bar_ds), use_container_width=True)
        st.caption(
            f"Total {len(ds):,} sampel. Dipakai untuk melatih model biner & "
            "multiclass di `src/latih_ai.py`."
        )
