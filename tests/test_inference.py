import csv
import json
from pathlib import Path

import numpy as np
import pytest

from odor_app.artifacts import BUNDLE, BundleError, load_bundle, manifest_digest
from odor_app.batch import batch_csv, predict_csv, read_smiles_csv, single_csv


def test_all_107_scores_match_original_models(predictor):
    fixtures = json.loads(Path(__file__).with_name("prediction_fixtures.json").read_text(encoding="utf-8"))
    for item in fixtures:
        result = predictor.predict(item["smiles"])
        scores = dict(result.scores)
        actual = np.asarray([scores[label] for label in predictor.bundle.labels])
        np.testing.assert_allclose(actual, item["scores"], rtol=1e-5, atol=1e-6)
        assert len(result.scores) == 107 and result.threshold == 0.6
        assert list(scores.values()) == sorted(scores.values(), reverse=True)
        np.testing.assert_array_equal(actual >= .6, np.asarray(item["scores"]) >= .6)
        assert len(list(csv.DictReader(single_csv(result).decode("utf-8-sig").splitlines()))) == 107


def test_manifest_rejects_wrong_digest():
    assert manifest_digest()
    with pytest.raises(BundleError):
        load_bundle(BUNDLE, "wrong-digest")


def test_batch_preserves_row_order_and_keeps_valid_rows(predictor):
    values = read_smiles_csv(b"smiles\nCCO\nC1\nCCO.O\nCC(=O)OCC\n")
    results = predict_csv(predictor, values)
    assert [result.row for result in results] == [1, 2, 3, 4]
    assert [result.prediction is not None for result in results] == [True, False, False, True]
    rows = list(csv.DictReader(batch_csv(results, predictor.bundle.labels).decode("utf-8-sig").splitlines()))
    assert [r["status"] for r in rows] == ["ok", "invalid", "invalid", "ok"]
    single_scores = dict(predictor.predict("CCO").scores)
    for label in predictor.bundle.labels:
        assert float(rows[0][f"skor_{label}"]) == single_scores[label]


@pytest.mark.parametrize(
    "raw", [b"id,smiles\n1,CCO\n", b"smiles\n", b"smiles\n" + b"CCO\n" * 101,
            b"X" * (1024 * 1024 + 1), b"smiles\n\xff\n", b'smiles\n"CCO'],
    ids=["header", "empty", "too-many-rows", "too-many-bytes", "encoding", "unclosed-quote"],
)
def test_invalid_batch_inputs(raw):
    with pytest.raises(ValueError):
        read_smiles_csv(raw)
