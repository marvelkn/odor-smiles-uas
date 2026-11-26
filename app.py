from time import perf_counter

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import streamlit as st

from odor_app.batch import MAX_BATCH_ROWS, batch_csv, make_csv, predict_csv, read_smiles_csv, single_csv
from odor_app.features import InvalidSmiles, MAX_SMILES_LENGTH
from odor_app.ui import get_predictor, log_event, public_data

st.set_page_config(page_title="Prediksi aroma molekul", page_icon="🧪", layout="centered")
st.title("Prediksi aroma molekul")
st.write("Masukkan struktur molekul dalam format SMILES untuk melihat kandidat label aroma dari model XGBoost.")
st.caption("Prototipe UAS IS411 · XGBoost Optuna · 107 label aroma")

examples = {item["name"]: item["smiles"] for item in public_data("examples.json")}


def pick_example():
    if choice := st.session_state.get("example"):
        st.session_state["smiles"] = examples[choice]
    st.session_state.pop("prediction", None)


def clear_single():
    st.session_state["example"] = None
    st.session_state["smiles"] = ""
    st.session_state.pop("prediction", None)


def clear_batch():
    st.session_state.pop("batch_result", None)
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1


def show_scores(rows):
    st.table([{"Label": label, "Skor model": f"{score:.4f}"} for label, score in rows])


def show_structure(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        drawer = rdMolDraw2D.MolDraw2DCairo(460, 250)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        st.image(drawer.GetDrawingText(), width=460, caption="Struktur 2D dari SMILES kanonik")


st.info("Gunakan struktur publik atau contoh bawaan. Input diproses di server Streamlit dan tidak ditulis ke penyimpanan aplikasi.")
single_tab, batch_tab = st.tabs(["Satu molekul", "Banyak molekul (CSV)"])

with single_tab:
    st.selectbox("Coba contoh molekul", list(examples), index=None, key="example",
                 placeholder="Pilih satu contoh", on_change=pick_example)
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
        st.caption(f"Model {prediction.model_version} · ambang {prediction.threshold:.2f}")
        st.text(prediction.smiles)
        st.markdown("**Label yang lolos ambang**")
        if prediction.passing:
            show_scores(prediction.passing)
        else:
            st.info("Tidak ada label yang mencapai ambang 0,60.")
        st.markdown("**Top 5 kandidat menurut skor**")
        show_scores(prediction.scores[:5])
        st.caption("Skor membantu memeringkat label. Skor belum dikalibrasi sebagai peluang persepsi manusia dan tidak mengukur intensitas aroma.")
        with st.expander("Lihat struktur molekul"):
            try:
                show_structure(prediction.smiles)
            except Exception:
                st.info("Gambar struktur belum tersedia. Skor prediksi tetap bisa digunakan.")
        st.download_button("Unduh skor 107 label", single_csv(prediction), "prediksi-aroma.csv", "text/csv")

with batch_tab:
    st.write(f"Unggah CSV UTF-8 berisi satu kolom `smiles`. Batas 1 MB dan {MAX_BATCH_ROWS} baris data.")
    st.download_button("Unduh template CSV", make_csv(("smiles",), ((value,) for value in examples.values())),
                       "template-smiles.csv", "text/csv")
    upload_version = st.session_state.get("upload_version", 0)
    with st.form(f"batch_{upload_version}"):
        upload = st.file_uploader("Pilih CSV", type=["csv"], max_upload_size=1, key=f"csv_{upload_version}")
        run_batch = st.form_submit_button("Prediksi CSV", type="primary")
    st.button("Bersihkan unggahan dan hasil", on_click=clear_batch)

    if run_batch:
        st.session_state.pop("batch_result", None)
        if upload is None:
            st.warning("Pilih berkas CSV terlebih dahulu.")
        else:
            try:
                values = read_smiles_csv(upload.getvalue())
            except ValueError as error:
                st.warning(str(error))
            else:
                start = perf_counter()
                try:
                    predictor = get_predictor()
                    with st.spinner(f"Memproses {len(values)} baris…"):
                        results = predict_csv(predictor, values)
                    st.session_state["batch_result"] = (results, predictor.bundle.labels)
                    log_event("batch", "ok", perf_counter() - start, len(values), predictor.bundle.version)
                except Exception:
                    st.error("CSV belum bisa diproses. Coba ulang dengan contoh dari template.")
                    log_event("batch", "unavailable", perf_counter() - start, len(values))

    if batch := st.session_state.get("batch_result"):
        results, labels = batch
        successful = sum(item.prediction is not None for item in results)
        st.subheader("Hasil CSV")
        st.write(f"{successful} dari {len(results)} baris berhasil diprediksi.")
        display = []
        for item in results:
            result = item.prediction
            display.append({"Baris data": item.row,
                            "Kandidat tertinggi": result.scores[0][0] if result else "—",
                            "Skor model": f"{result.scores[0][1]:.4f}" if result else "—",
                            "Lolos ambang": str(len(result.passing)) if result else "—",
                            "Status": "Berhasil" if result else item.message})
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption("Urutan dihitung dari baris data pertama setelah header. Unduhan memuat skor 107 label untuk setiap baris yang berhasil.")
        st.download_button("Unduh hasil CSV", batch_csv(results, labels), "hasil-prediksi-aroma.csv", "text/csv")
