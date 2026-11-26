from time import perf_counter

from rdkit import Chem
import streamlit as st

from odor_app.batch import MAX_BATCH_ROWS, batch_csv, make_csv, predict_csv, read_smiles_csv, single_csv
from odor_app.catalog import TOP_CANDIDATES, show_catalog, show_comparison
from odor_app.features import InvalidSmiles, MAX_SMILES_LENGTH
from odor_app.ui import get_predictor, log_event, public_data

st.set_page_config(page_title="Prediksi aroma molekul", page_icon="🧪", layout="centered")
st.title("Prediksi aroma molekul")
st.write("Pilih molekul dari katalog atau masukkan SMILES sendiri untuk melihat kandidat label aroma.")
st.caption("Prototipe UAS IS411 · XGBoost Optuna · 107 label aroma")

catalog = public_data("molecule_catalog.json")
catalog_by_name = {item["name"]: item for item in catalog}
catalog_smiles = [item["smiles"] for item in catalog]


def pick_example():
    if choice := st.session_state.get("example"):
        st.session_state["smiles"] = catalog_by_name[choice]["smiles"]
    st.session_state.pop("prediction", None)


def clear_single():
    st.session_state["example"] = None
    st.session_state["smiles"] = ""
    st.session_state.pop("prediction", None)


def clear_batch():
    st.session_state.pop("batch_outcome", None)
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1


def show_structure(smiles):
    # Keep prediction available even if the host cannot initialize the drawing backend.
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        drawer = rdMolDraw2D.MolDraw2DCairo(460, 250)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        st.image(drawer.GetDrawingText(), width=460, caption="Struktur 2D dari SMILES kanonik")


st.info("Gunakan struktur publik atau contoh bawaan. Input diproses di server Streamlit dan tidak ditulis ke penyimpanan aplikasi.")
single_tab, compare_tab, catalog_tab, batch_tab = st.tabs(
    ["Satu molekul", "Bandingkan", "Katalog", "Banyak molekul (CSV)"]
)

with single_tab:
    chosen = st.selectbox(f"Pilih dari katalog {len(catalog)} molekul", list(catalog_by_name), index=None,
                          key="example", placeholder="Pilih atau cari nama molekul", on_change=pick_example)
    if chosen:
        cid = catalog_by_name[chosen]["cid"]
        st.caption(f"Pilihan katalog: {chosen} · [PubChem CID {cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
    else:
        st.caption("Memilih nama akan mengisi SMILES otomatis. Tiga contoh awal: Etanol, Etil asetat, dan Vanilin.")
    with st.form("single"):
        smiles = st.text_input("SMILES", placeholder="Contoh: CCO", key="smiles", max_chars=MAX_SMILES_LENGTH)
        submit = st.form_submit_button("Prediksi aroma", type="primary")
    st.button("Bersihkan input dan hasil", on_click=clear_single)

    if submit:
        st.session_state.pop("prediction", None)
        start = perf_counter()
        try:
            if not smiles.strip():
                raise InvalidSmiles("empty", "Masukkan SMILES atau pilih contoh molekul.")
            predictor = get_predictor()
            with st.spinner("Menghitung skor aroma…"):
                prediction = predictor.predict(smiles)
            st.session_state["prediction"] = prediction
            log_event("predict", "ok", perf_counter() - start, 1, prediction.model_version)
        except InvalidSmiles as error:
            st.warning(str(error))
            log_event("predict", error.code, perf_counter() - start, 1)
        except Exception:
            st.error("Prediksi belum tersedia. Muat ulang halaman atau coba lagi beberapa saat.")
            log_event("predict", "unavailable", perf_counter() - start, 1)

    if prediction := st.session_state.get("prediction"):
        st.subheader("Hasil prediksi")
        threshold = f"{prediction.threshold:.2f}".replace(".", ",")
        st.caption(f"Model {prediction.model_version}")
        st.text(prediction.smiles)
        passed_count = len(prediction.passing)
        st.write(f"**{passed_count} dari {len(prediction.scores)} label** lolos ambang {threshold}.")
        st.table([
            {
                "Label aroma": label,
                "Skor model": f"{score:.4f}",
                "Status": "Lolos ambang" if score >= prediction.threshold else "Di bawah ambang",
            }
            for label, score in prediction.scores[:max(passed_count, TOP_CANDIDATES)]
        ])
        st.caption("Skor membantu memeringkat label. Skor belum dikalibrasi sebagai peluang persepsi manusia dan tidak mengukur intensitas aroma.")
        with st.expander("Lihat struktur molekul"):
            try:
                show_structure(prediction.smiles)
            except Exception:
                st.info("Gambar struktur belum tersedia. Skor prediksi tetap bisa digunakan.")
        st.download_button("Unduh skor 107 label", single_csv(prediction), "prediksi-aroma.csv", "text/csv", on_click="ignore")

with compare_tab:
    show_comparison(catalog_by_name)

with catalog_tab:
    show_catalog(catalog)

with batch_tab:
    st.write(f"Jalankan prediksi untuk {len(catalog)} molekul dari katalog, atau unggah CSV kamu sendiri.")
    use_catalog = st.button(f"Prediksi {len(catalog)} molekul katalog", type="primary", key="run_catalog")
    template_col, catalog_col = st.columns(2)
    template_col.download_button("Unduh template CSV", make_csv(("smiles",), ((value,) for value in catalog_smiles[:3])),
                                 "template-smiles.csv", "text/csv", on_click="ignore")
    catalog_col.download_button("Unduh CSV katalog", make_csv(("smiles",), ((value,) for value in catalog_smiles)),
                                "katalog-smiles.csv", "text/csv", on_click="ignore")
    st.caption(f"Format unggahan: CSV UTF-8 dengan satu kolom `smiles`, maksimal 1 MB dan {MAX_BATCH_ROWS} baris data.")
    upload_version = st.session_state.get("upload_version", 0)
    with st.form(f"batch_{upload_version}"):
        upload = st.file_uploader("Pilih CSV", type=["csv"], max_upload_size=1, key=f"csv_{upload_version}")
        run_batch = st.form_submit_button("Prediksi CSV", type="primary")
    st.button("Bersihkan unggahan dan hasil", on_click=clear_batch)

    if use_catalog or run_batch:
        st.session_state.pop("batch_outcome", None)
        try:
            if use_catalog:
                values = catalog_smiles
            elif upload is None:
                raise ValueError("Pilih berkas CSV terlebih dahulu.")
            else:
                values = read_smiles_csv(upload.getvalue())
        except ValueError as error:
            st.warning(str(error))
        else:
            start = perf_counter()
            try:
                predictor = get_predictor()
                with st.spinner(f"Memproses {len(values)} molekul…"):
                    results = predict_csv(predictor, values)
                names = list(catalog_by_name) if use_catalog else None
                st.session_state["batch_outcome"] = (results, predictor.bundle.labels, names)
                log_event("batch_catalog" if use_catalog else "batch", "ok", perf_counter() - start,
                          len(values), predictor.bundle.version)
            except Exception:
                st.error("Prediksi batch belum tersedia. Muat ulang halaman atau coba lagi beberapa saat.")
                log_event("batch_catalog" if use_catalog else "batch", "unavailable", perf_counter() - start, len(values))

    if batch := st.session_state.get("batch_outcome"):
        results, labels, names = batch
        successful = sum(item.prediction is not None for item in results)
        st.subheader("Hasil katalog" if names else "Hasil CSV")
        st.write(f"{successful} dari {len(results)} baris berhasil diprediksi.")
        display = []
        for item in results:
            result = item.prediction
            identity = {"Molekul": names[item.row - 1]} if names else {"Baris data": item.row}
            display.append({**identity,
                            "Kandidat tertinggi": result.scores[0][0] if result else "—",
                            "Skor model": f"{result.scores[0][1]:.4f}" if result else "—",
                            "Lolos ambang": str(len(result.passing)) if result else "—",
                            "Status": "Berhasil" if result else item.message})
        st.dataframe(display, hide_index=True, width="stretch")
        if not names:
            st.caption("Urutan dihitung dari baris data pertama setelah header.")
        st.caption("Unduhan memuat skor 107 label untuk setiap molekul yang berhasil diprediksi.")
        st.download_button("Unduh hasil CSV", batch_csv(results, labels, names=names),
                           "hasil-prediksi-aroma.csv", "text/csv", on_click="ignore")
