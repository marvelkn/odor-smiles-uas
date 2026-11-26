import streamlit as st

from odor_app.ui import public_data

st.set_page_config(page_title="Metode dan evaluasi", page_icon="🧪", layout="centered")
st.title("Metode dan evaluasi")
evidence = public_data("evaluation.json")

st.markdown(
    "Aplikasi ini memakai **XGBoost Optuna** dari run UAS `uas_gpu_lanjutan`. "
    "Model dipilih berdasarkan **Macro AP tertinggi pada lima fold data training**. "
    "Model mengolah **2.048 bit Morgan** dan **8 deskriptor RDKit**, lalu memakai mask dari training. "
    "Satu molekul dapat memperoleh skor untuk **107 label**."
)

st.subheader("Perbandingan kandidat Optuna")
st.table([
    {"Model": row["model"], "Macro AP CV training": f"{row['cv_macro_ap']:.5f}",
     "Macro AP test historis": f"{row['ap_macro']:.5f}"} for row in evidence["optuna"]
])
st.write("LightGBM Optuna memperoleh Macro AP test tertinggi pada split ini. Pemilihan model aplikasi mengikuti aturan CV training yang dicatat sebelum evaluasi run final.")
st.caption("Macro AP merata-ratakan kualitas pemeringkatan positif per label. Nilai 0,23405 tidak berarti akurasi 23,4%.")

st.subheader("Data dan batas penggunaan")
st.markdown(
    "Data proyek berasal dari tabel kurasi [Suh, Hong, dan Park (2025)](https://doi.org/10.1038/s42004-025-01651-7). "
    "Cleaning menghasilkan **8.182 molekul unik**. Proyek memakai preprocessing, fitur, dan evaluasi sendiri. "
    "[Data dan kode sumber penulis](https://osf.io/vfru6/)."
)
catalog_size = len(public_data("molecule_catalog.json"))
st.write(f"Katalog berisi {catalog_size} molekul dari data proyek yang sudah dibersihkan. "
         "Nama diperiksa melalui kecocokan CID dan struktur SMILES. Katalog disediakan sebagai pilihan input untuk demo.")
st.markdown(
    "Data test telah dipakai pada tahap sebelumnya, sehingga evaluasinya **eksploratif**. "
    "Anotasi yang kosong belum tentu berarti aroma itu tidak hadir. "
    "Skor yang ditampilkan adalah **skor pemeringkatan model**, belum terkalibrasi menjadi peluang persepsi atau ukuran intensitas."
)
st.write("Aplikasi memproses struktur publik untuk penyaringan awal. Gunakan versi lokal untuk struktur yang belum boleh dibagikan ke layanan cloud.")
