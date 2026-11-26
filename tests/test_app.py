import csv
import json
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

from odor_app.batch import MAX_BATCH_ROWS, batch_csv, make_csv, read_smiles_csv
from odor_app.inference import Prediction
from odor_app import ui

ROOT = Path(__file__).resolve().parents[1]


def predict_button(app):
    return next(button for button in app.button if button.label == "Prediksi aroma")


def test_example_prediction_and_clear():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.selectbox(key="example").options) == 500
    app.selectbox(key="example").select("Etanol").run()
    assert app.text_input(key="smiles").value == "CCO"
    predict_button(app).click().run()
    assert not app.exception and not app.error
    assert any(item.value == "Hasil prediksi" for item in app.subheader)
    assert any(len(table.value) == 5 for table in app.table)
    next(button for button in app.button if button.label == "Bersihkan input dan hasil").click().run()
    assert app.text_input(key="smiles").value == ""
    assert not any(item.value == "Hasil prediksi" for item in app.subheader)


def test_prediction_from_catalog_without_uploading():
    catalog = json.loads((ROOT / "data/molecule_catalog.json").read_text(encoding="utf-8"))
    assert len(catalog) == MAX_BATCH_ROWS == 500
    for key in ("name", "smiles", "cid"):
        assert len({item[key] for item in catalog}) == 500
    all_smiles = [item["smiles"] for item in catalog]
    assert read_smiles_csv(make_csv(("smiles",), zip(all_smiles))) == all_smiles

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.button(key="run_catalog").click().run()
    assert not app.exception and not app.error
    assert any(item.value == "Hasil katalog" for item in app.subheader)
    results, labels, names = app.session_state["batch_outcome"]
    assert len(results) == 500 and all(item.prediction is not None for item in results)
    batch_table = next(item.value for item in app.dataframe if "Kandidat tertinggi" in item.value.columns)
    assert batch_table["Molekul"].tolist() == names == [item["name"] for item in catalog]
    exported = batch_csv(results, labels, names=names)
    rows = list(csv.DictReader(exported.decode("utf-8-sig").splitlines()))
    assert len(rows) == 500 and rows[0]["molekul"] == "Etanol" and rows[0]["status"] == "ok"
    assert rows[-1]["molekul"] == names[-1] and rows[-1]["status"] == "ok"


def test_catalog_searches_by_name_cid_and_smiles():
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    for query in ("vAniLiN", "1183", "COc1cc(C=O)ccc1O"):
        app.text_input(key="catalog_search").set_value(query).run()
        assert not app.exception
        assert "Vanilin" in app.dataframe[0].value["Molekul"].tolist()
    app.text_input(key="catalog_search").set_value("nama yang tidak ada").run()
    assert any("Tidak ada hasil" in item.value for item in app.info)


def test_compare_displays_scores_per_molecule_and_resets_for_new_selection():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert app.button(key="run_comparison").disabled
    names = ["Etanol", "Vanilin", "Etil asetat"]
    app.multiselect(key="compare_names").set_value(names).run()
    app.button(key="run_comparison").click().run()
    assert not app.exception and not app.error
    stored_names, predictions = app.session_state["catalog_comparison"]
    assert stored_names == tuple(names)
    table = next(item.value for item in app.dataframe if "Label aroma" in item.value.columns)
    assert list(table.columns) == ["Label aroma", *names]
    assert set(table["Label aroma"]) == {label for pred in predictions for label, _ in pred.scores[:5]}
    for name, pred in zip(names, predictions, strict=True):
        assert table[name].tolist() == [dict(pred.scores)[label] for label in table["Label aroma"]]
    app.multiselect(key="compare_names").set_value(names[:2]).run()
    assert not any(item.value == "Hasil perbandingan" for item in app.subheader)


def test_invalid_smiles_has_actionable_message():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.text_input(key="smiles").set_value("CCO.O")
    predict_button(app).click().run()
    assert not app.exception
    assert any("multikomponen" in item.value for item in app.warning)


def test_missing_drawing_backend_does_not_block_prediction(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit.Chem.Draw", None)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.text_input(key="smiles").set_value("CCO")
    predict_button(app).click().run()
    assert not app.exception and not app.error
    assert any(item.value == "Hasil prediksi" for item in app.subheader)
    assert any("Gambar struktur belum tersedia" in item.value for item in app.info)


def test_zero_labels_pass_does_not_hide_ranked_candidates(monkeypatch):
    class LowScorePredictor:
        def predict(self, smiles):
            return Prediction(smiles, tuple((f"LABEL_{n}", .45 - n / 100) for n in range(5)), .6, "fixture")

    monkeypatch.setattr(ui, "get_predictor", LowScorePredictor)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    app.text_input(key="smiles").set_value("CCO")
    predict_button(app).click().run()
    assert not app.error and not app.exception
    assert any("Tidak ada label" in item.value for item in app.info)
    assert len(app.table[0].value) == 5


def test_evaluation_shows_both_training_and_test():
    app = AppTest.from_file(str(ROOT / "pages/1_Metode_dan_Evaluasi.py")).run()
    assert not app.exception
    assert list(app.table[0].value.columns) == ["Model", "Macro AP CV training", "Macro AP test historis"]
