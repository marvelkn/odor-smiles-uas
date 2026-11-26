"""Export trusted lab checkpoints and verify CPU scores against saved test predictions.

Run from the deploy repository: python -m scripts.build_bundle /path/to/paket-lab
"""

import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from scipy import sparse

from odor_app.artifacts import BUNDLE, load_bundle, manifest_digest
from odor_app.inference import Predictor

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "xgboost_optuna"
RUN = "uas_gpu_lanjutan"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def check(folder, name, expected):
    raw = (folder / name).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Checksum sumber berbeda: {name}")
    return raw


def build(package: Path):
    if BUNDLE.exists():
        raise FileExistsError(f"Pertahankan bundle sebelumnya: {BUNDLE}")
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    run = package / "Hasil/hasil_eksperimen" / RUN
    source = run / "candidates" / CANDIDATE
    prepared = package / "prepared_v2"
    evaluation = run / "evaluation" / CANDIDATE
    frozen = read_json(run / "selection_frozen.json")["selection"]
    assert frozen["selected_candidate"] == CANDIDATE
    fit = json.loads(check(source, "fit_complete.json", frozen["candidates"][CANDIDATE]["fit_complete_sha256"]))["files"]
    recipe = json.loads(check(source, "recipe.json", fit["recipe.json"]))
    selection = json.loads(check(source, "selection_metrics.json", fit["selection_metrics.json"]))
    assert selection["threshold_source"] == "validation" and selection["selected_threshold"] == .6
    assert recipe["features"] == "combined"
    spec = read_json(prepared / "manifest.json")
    prepared_hashes = spec["outputs_sha256"]
    labels = json.loads(check(prepared, "label_names.json", prepared_hashes["label_names.json"]))
    assert recipe["labels"] == labels and len(labels) == 107
    completion = read_json(run / "pipeline_complete.json")["files"]
    summary = check(run, "summary.csv", completion["summary.csv"])
    check(evaluation, "probabilities.npz", read_json(evaluation / "complete.json")["files"]["probabilities.npz"])
    for filename in ("X_test.npz", "row_ids_test.npy", "molecules.csv"):
        check(prepared, filename, prepared_hashes[filename])
    examples = read_json(ROOT / "data/examples.json")

    with TemporaryDirectory(dir=BUNDLE.parent, prefix=".bundle-build-") as temp:
        staging = Path(temp) / BUNDLE.name
        (staging / "models").mkdir(parents=True)
        (staging / "feature_mask.npy").write_bytes(check(source, "feature_mask.npy", fit["feature_mask.npy"]))
        write_json(staging / "labels.json", labels)
        models = []
        reference_models = []
        for index, label in enumerate(labels):
            original = f"models/label_{index:03d}.joblib"
            check(source, original, fit[original])
            # Only trusted, verified artifacts from this local project may be unpickled.
            model = joblib.load(source / original)
            if isinstance(model, dict) and "constant" in model:
                reference_models.append(float(model["constant"]))
                models.append({"label": label, "constant": float(model["constant"])})
                continue
            assert list(model.classes_) == [0, 1]
            model.set_params(device="cpu", n_jobs=1)
            reference_models.append(model)
            output = f"models/label_{index:03d}.ubj.gz"
            to_save = model.get_booster().save_raw(raw_format="ubj")
            (staging / output).write_bytes(gzip.compress(to_save, compresslevel=6, mtime=0))
            models.append({"label": label, "path": output})

        hashes = {file.relative_to(staging).as_posix(): sha256(file.read_bytes()).hexdigest()
                  for file in sorted(staging.rglob("*")) if file.is_file()}
        feature_spec = spec["feature_spec"]
        write_json(staging / "manifest.json", {
            "version": BUNDLE.name, "format_version": 1, "candidate": CANDIDATE,
            "source_run": RUN, "threshold": selection["selected_threshold"],
            "training_macro_ap_cv": selection["cv_macro_ap"],
            "source_selection_sha256": sha256((run / "selection_frozen.json").read_bytes()).hexdigest(),
            "feature_spec": {key: feature_spec[key] for key in (
                "morgan_bits", "morgan_radius", "include_chirality", "descriptors"
            )} | {"dtype": feature_spec["descriptor_dtype"]},
            "format": "gzip + native XGBoost UBJSON, one file per label",
            "models": models, "sha256": hashes,
        })

        predictor = Predictor(load_bundle(staging, manifest_digest(staging)))
        features = np.ascontiguousarray(sparse.load_npz(prepared / "X_test.npz").toarray(), dtype=np.float32)
        row_ids = np.load(prepared / "row_ids_test.npy", allow_pickle=False)
        with np.load(evaluation / "probabilities.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(row_ids, saved["row_ids"])
            expected = saved["probabilities"]
        actual = predictor.score_rows(features)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(actual >= .6, expected >= .6)

        # Also cover the real path from SMILES, not just the precomputed test matrix.
        with (prepared / "molecules.csv").open(encoding="utf-8", newline="") as handle:
            smiles_by_id = {int(row["molecule_id"]): row["smiles"] for row in csv.DictReader(handle)}
        indices = np.linspace(0, len(row_ids) - 1, 16, dtype=int)
        regenerated = np.stack([predictor.features.transform(smiles_by_id[int(row_ids[i])])[1] for i in indices])
        np.testing.assert_array_equal(regenerated, features[indices])
        np.testing.assert_allclose(predictor.score_rows(regenerated), expected[indices], rtol=1e-5, atol=1e-6)

        smoke_features = np.stack([predictor.features.transform(item["smiles"])[1] for item in examples])
        masked = np.ascontiguousarray(smoke_features[:, predictor.bundle.mask])
        reference = np.column_stack([
            np.full(len(masked), model) if isinstance(model, float) else model.predict_proba(masked)[:, 1]
            for model in reference_models
        ])
        np.testing.assert_allclose(predictor.score_rows(smoke_features), reference, rtol=1e-6, atol=1e-7)
        fixtures = [{"smiles": item["smiles"], "scores": row.tolist()}
                    for item, row in zip(examples, reference, strict=True)]
        metrics = []
        names = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "random_forest": "Random Forest"}
        for row in csv.DictReader(summary.decode("utf-8").splitlines()):
            if row["condition"] == "optuna":
                metrics.append({"model": names[row["family"]], "cv_macro_ap": float(row["cv_macro_ap"]),
                                "ap_macro": float(row["ap_macro"])})
        staging.rename(BUNDLE)

    write_json(ROOT / "data/evaluation.json", {"scope": "retrospective_reused_test", "optuna": metrics})
    write_json(ROOT / "tests/prediction_fixtures.json", fixtures)
    print(f"Verified: {len(row_ids)} × {len(labels)} scores, identical decisions at 0.60; "
          f"max absolute GPU→CPU deviation {float(np.max(np.abs(actual - expected))):.3g}.")
    print(f"Reproduced features exactly for {len(indices)} test SMILES; "
          f"saved {len(models)} portable models, {sum(x.stat().st_size for x in BUNDLE.rglob('*') if x.is_file()):,} bytes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    build(parser.parse_args().package.resolve())
