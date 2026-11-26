# Model card · XGBoost Optuna UAS IS411

| Item | Keterangan |
|---|---|
| Versi bundle | `xgboost-optuna-v1` |
| Sumber | Run UAS `uas_gpu_lanjutan`, kandidat `xgboost_optuna` |
| Seleksi | Macro AP tertinggi pada 5-fold CV training, **0,23405** |
| Test historis | XGBoost Optuna **0,22965**. LightGBM Optuna memperoleh Macro AP test tertinggi, **0,23260**. |
| Input | SMILES satu molekul, yang dikanonisasi menjadi isomeric SMILES |
| Fitur | Morgan radius 2, 2.048 bit, tanpa kiralitas. Delapan deskriptor RDKit. Mask run menyisakan 2.051 fitur. |
| Output | 107 skor, dengan ambang global **0,60** yang dipilih pada validation |
| Implementasi | XGBoost `2.1.4` melayani prediksi di CPU. Model diekspor ke native UBJSON tanpa fit ulang. |

## Data dan atribusi

Suh, J., Hong, Y., & Park, C. (2025). *A comparative study of machine learning models on molecular fingerprints for odor decoding*. Communications Chemistry, 8, 278. [Artikel](https://doi.org/10.1038/s42004-025-01651-7) · [Arsip OSF](https://osf.io/vfru6/).

Proyek menggunakan dataset kurasi penulis sebagai masukan. Tahap proyek mencakup validasi struktur, penggabungan duplikat, pembuatan fitur, split grup, seleksi label dan pelatihan ulang pada data proyek. Cleaning menghasilkan **8.182 molekul unik**. Aplikasi mendistribusikan bundle model hasil kerja tim, contoh publik, dan ringkasan evaluasi. Dataset mentah dan prediksi historis per molekul tidak disertakan.

Manifest proyek mencatat [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) untuk dataset sumber OSF versi 1. Rilis aplikasi mencantumkan sumber dan perubahan preprocessing di atas.

## Verifikasi portabilitas

Pada CPU, bundle native menghasilkan skor yang cocok dengan prediksi tersimpan pada 1.657 molekul test × 107 label, dengan selisih absolut maksimum **1,19 × 10⁻⁷**. Keputusan pada ambang 0,60 identik untuk seluruh pasangan. Fitur dari 16 SMILES test juga direproduksi persis. Pemeriksaan ini memvalidasi implementasi inferensi dan tidak menambah bukti generalisasi model.

## Keterbatasan

- Test historis sudah pernah digunakan sebelum run final, sehingga kinerja yang dilaporkan bersifat eksploratif. Tidak ada estimasi generalisasi dari test eksternal independen dalam rilis ini.
- Label yang tidak tercatat dalam kurasi belum tentu merupakan negatif sensori. Label dengan sedikit contoh positif memiliki estimasi yang lebih tidak stabil.
- Pembagian berdasarkan grup mengurangi duplikasi struktur lintas split, tetapi analog dekat masih mungkin tersebar. Fingerprint tidak mengodekan kiralitas.
- Skor memeringkat anotasi kandidat. Gunakan pengujian sensori untuk mengonfirmasi aroma dan jangan gunakan skor sebagai ukuran intensitas atau keamanan.

## Akses dan privasi

UI menerima input tunggal dan batch kecil. Prediksi dilakukan di proses Streamlit. Input dan hasil pengguna tidak ditulis ke log aplikasi atau ke penyimpanan permanen. Struktur rahasia sebaiknya diproses dengan instalasi lokal.
