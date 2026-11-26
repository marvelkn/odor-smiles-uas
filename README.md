# Prediksi aroma molekul · UAS IS411

Aplikasi Streamlit untuk memeringkat 107 label aroma dari satu struktur molekul dalam format SMILES. Aplikasi memuat model XGBoost Optuna dari penelitian UAS pada CPU. Pilih salah satu dari 100 molekul katalog, tempel SMILES sendiri, atau jalankan prediksi batch.

## Jalankan lokal

Gunakan Python 3.12. Jalankan perintah berikut dari direktori yang berisi `app.py`.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Buka `http://localhost:8501`, pilih contoh **Etanol**, lalu klik **Prediksi aroma**. Hasil memisahkan label yang mencapai ambang `0,60` dari lima kandidat dengan skor tertinggi. Unduhan CSV memuat skor semua label dan identitas versi model.

Untuk batch, klik **Prediksi 100 molekul katalog** agar seluruh contoh langsung diproses. Kamu juga bisa mengunduh **CSV katalog** atau **template CSV**. Aplikasi menerima CSV UTF-8 dengan satu kolom `smiles`, maksimal 100 baris dan 1 MB. Baris dengan struktur yang tidak valid mendapat pesan sendiri, sementara baris valid tetap diproses.

Katalog di [data/molecule_catalog.json](data/molecule_catalog.json) adalah 100 struktur unik dari data UAS yang sudah dibersihkan. Nama dicocokkan dari tabel sumber dengan CID dan SMILES kanonik. Katalog menyediakan contoh input untuk mencoba aplikasi dan tidak digunakan untuk melaporkan hasil evaluasi baru.

## Deploy ke Streamlit Community Cloud

1. Publikasikan direktori ini sebagai repo GitHub yang berisi bundle native di `artifacts/xgboost-optuna-v1/`. Pertahankan file model terpisah.
2. Pastikan tab **Actions** menunjukkan tes CPU dan aplikasi lulus.
3. Buka [Streamlit Community Cloud](https://share.streamlit.io/) dan pilih **Create app**. Gunakan repo yang baru, branch `main`, serta `app.py` sebagai entrypoint.
4. Pada **Advanced settings**, pilih **Python 3.12** atau **Python 3.14**. `requirements.txt` memilih versi SciPy yang sesuai dengan Python pada host. Bundle ada di repo sehingga tidak membutuhkan secret.
5. Deploy, buka URL publik, dan uji contoh bawaan, input yang salah, unduhan, serta CSV campuran valid dan invalid. Ulangi dari perangkat atau sesi lain.
6. Jalankan satu prediksi sebelum presentasi karena layanan dapat berhibernasi setelah tidak digunakan. Siapkan versi lokal sebagai cadangan.

Panduan resmi: [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) dan [batas resource dan hibernasi](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-hibernation).

### Docker

Untuk host yang mendukung Docker atau demo lokal, gunakan perintah berikut. Gunakan hasil pengukuran aktual untuk memilih kapasitas host.

```bash
docker build -t odor-smiles-uas .
docker run --rm -p 8501:8501 odor-smiles-uas
```

## Tes dan verifikasi

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m scripts.benchmark
```

Skrip benchmark melaporkan waktu pemuatan native, p95 inferensi lokal, satu batch 100 molekul, memori proses, serta konsistensi dua worker. Hasil lokal memberi baseline untuk dibandingkan dengan pengukuran di cloud.

Jika tim perlu membangun ulang bundle dari paket eksperimen, pasang `requirements-export.txt` di environment Python terpisah dan jalankan `python -m scripts.build_bundle /path/to/Run_GPU_Lanjutan` di checkout yang belum memiliki bundle. Skrip memeriksa checksum sumber, mengekspor format native, dan membandingkan semua skor test dengan hasil tersimpan. Gunakan hanya checkpoint `.joblib` lokal dari paket lab terpercaya. Aplikasi yang dipublikasikan memakai file UBJSON terkompresi dan tidak memerlukan `joblib` saat inferensi. Environment aplikasi memakai NumPy 2.3.5 untuk kompatibilitas dengan PyArrow yang digunakan tabel Streamlit. Environment ekspor mempertahankan NumPy 1.26.4 dari eksperimen.

## Batas penggunaan dan privasi

Gunakan aplikasi untuk demo dan penyaringan awal. Skor belum dikalibrasi sebagai peluang persepsi dan tidak mengukur intensitas aroma. Evaluasi proyek bersifat eksploratif karena test pernah dipakai pada tahap sebelumnya.

Saat menggunakan versi cloud, SMILES dan CSV dikirim ke server Streamlit untuk diproses. Aplikasi menyimpan hasil sementara dalam sesi aktif, tanpa database atau cache lintas pengguna untuk input. Log aplikasi memuat kategori status, durasi, jumlah baris, dan versi model. Gunakan aplikasi lokal untuk struktur yang belum boleh dibagikan.

Lihat [model card](MODEL_CARD.md) untuk pemilihan model, provenance, dan batas evaluasi.
