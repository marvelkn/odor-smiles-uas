"""Browse public examples and compare their model scores in a session."""

from time import perf_counter

import streamlit as st

from .batch import make_csv
from .ui import get_predictor, log_event

TOP_CANDIDATES = 5
MAX_COMPARE = 3


def show_catalog(catalog):
    st.subheader("Cari molekul")
    query = st.text_input(
        "Nama, CID, atau SMILES", placeholder="Contoh: Vanilin, 1183, atau CCO", key="catalog_search",
    ).strip()
    if query.isdecimal():
        matches = [entry for entry in catalog if str(entry["cid"]) == query]
    else:
        name_query = query.casefold()
        matches = [
            entry for entry in catalog
            if name_query in entry["name"].casefold() or query in entry["smiles"]
        ]
    st.caption(f"{len(matches)} dari {len(catalog)} molekul")
    if not matches:
        st.info("Tidak ada hasil. Coba sebagian nama molekul atau masukkan CID lengkap.")
        return
    st.dataframe(
        [{"Molekul": entry["name"], "PubChem": f"https://pubchem.ncbi.nlm.nih.gov/compound/{entry['cid']}",
          "SMILES": entry["smiles"]} for entry in matches],
        hide_index=True,
        width="stretch",
        column_config={"PubChem": st.column_config.LinkColumn("PubChem CID", display_text=r".*/(\d+)$")},
    )
    st.download_button(
        "Unduh daftar dengan nama dan CID",
        make_csv(("molekul", "cid", "smiles"), ((e["name"], e["cid"], e["smiles"]) for e in matches)),
        "daftar-molekul.csv",
        "text/csv",
        on_click="ignore",
    )


def show_comparison(catalog_by_name):
    st.subheader("Bandingkan prediksi aroma")
    selected = tuple(st.multiselect(
        "Pilih 2–3 molekul dari katalog", list(catalog_by_name), key="compare_names",
        max_selections=MAX_COMPARE, placeholder="Cari nama molekul",
    ))
    stored = st.session_state.get("catalog_comparison")
    if stored and stored[0] != selected:
        st.session_state.pop("catalog_comparison")
    if st.button("Bandingkan skor", type="primary", disabled=len(selected) < 2, key="run_comparison"):
        st.session_state.pop("catalog_comparison", None)
        start = perf_counter()
        try:
            predictor = get_predictor()
            with st.spinner("Menghitung skor aroma…"):
                features = [predictor.features.transform(catalog_by_name[name]["smiles"]) for name in selected]
                predictions = predictor.predict_many(features)
            st.session_state["catalog_comparison"] = selected, predictions
            log_event("compare_catalog", "ok", perf_counter() - start, len(selected), predictor.bundle.version)
        except Exception:
            st.error("Perbandingan belum tersedia. Coba lagi beberapa saat.")
            log_event("compare_catalog", "unavailable", perf_counter() - start, len(selected))

    if outcome := st.session_state.get("catalog_comparison"):
        names, predictions = outcome
        st.subheader("Hasil perbandingan")
        threshold = f"{predictions[0].threshold:.2f}".replace(".", ",")
        st.caption(
            f"Tabel menampilkan gabungan {TOP_CANDIDATES} kandidat teratas tiap molekul. "
            f"Ambang model {threshold}. Setiap batang memakai skala skor 0–1 yang sama."
        )
        scores = [dict(pred.scores) for pred in predictions]
        labels = {label for pred in predictions for label, _ in pred.scores[:TOP_CANDIDATES]}
        ranked = sorted(labels, key=lambda label: (-max(row[label] for row in scores), label))
        st.dataframe(
            [{"Label aroma": label, **{name: row[label] for name, row in zip(names, scores, strict=True)}}
             for label in ranked],
            hide_index=True,
            width="stretch",
            column_config={name: st.column_config.ProgressColumn(name, min_value=0, max_value=1, format="%.4f")
                           for name in names},
        )
        st.caption("Skor membantu memeringkat kandidat aroma. Skor belum dikalibrasi sebagai peluang persepsi atau ukuran intensitas.")
        st.download_button(
            "Unduh skor perbandingan",
            make_csv(
                ("molekul", "cid", "smiles_kanonik", "label", "skor_model", "lolos_ambang", "ambang", "versi_model"),
                ((name, catalog_by_name[name]["cid"], pred.smiles, label, score, score >= pred.threshold,
                  pred.threshold, pred.model_version)
                 for name, pred in zip(names, predictions, strict=True) for label, score in pred.scores),
            ),
            "perbandingan-aroma.csv",
            "text/csv",
            on_click="ignore",
        )
