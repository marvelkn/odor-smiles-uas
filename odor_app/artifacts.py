"""Verify and load only the app's own immutable model bundle."""

from dataclasses import dataclass
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
from xgboost import Booster

from .features import DESCRIPTORS, FEATURE_COUNT

BUNDLE = Path(__file__).resolve().parents[1] / "artifacts/xgboost-optuna-v1"


class BundleError(RuntimeError):
    """Missing or incompatible release artifacts."""


@dataclass(frozen=True)
class Bundle:
    version: str
    labels: tuple[str, ...]
    threshold: float
    mask: np.ndarray
    models: tuple[Booster | float, ...]


def manifest_digest(root: Path = BUNDLE) -> str:
    return sha256((root / "manifest.json").read_bytes()).hexdigest()


def load_bundle(root: Path, expected_manifest_digest: str) -> Bundle:
    raw_manifest = (root / "manifest.json").read_bytes()
    if sha256(raw_manifest).hexdigest() != expected_manifest_digest:
        raise BundleError("Manifest berubah selama pemuatan.")

    manifest = json.loads(raw_manifest)
    if (manifest["format_version"], manifest["candidate"], manifest["threshold"]) != (1, "xgboost_optuna", .6):
        raise BundleError("Identitas model atau ambang tidak sesuai.")
    if manifest["feature_spec"] != {
        "morgan_bits": 2048, "morgan_radius": 2, "include_chirality": False,
        "descriptors": list(DESCRIPTORS), "dtype": "float32",
    }:
        raise BundleError("Spesifikasi fitur tidak cocok.")

    def verified(relative: str) -> bytes:
        raw = (root / relative).read_bytes()
        if sha256(raw).hexdigest() != manifest["sha256"][relative]:
            raise BundleError(f"Checksum gagal untuk {relative}")
        return raw

    labels = tuple(json.loads(verified("labels.json")))
    mask = np.load(io.BytesIO(verified("feature_mask.npy")), allow_pickle=False)
    if len(labels) != 107 or len(set(labels)) != 107:
        raise BundleError("Bundle harus memiliki 107 label unik.")
    if mask.shape != (FEATURE_COUNT,) or mask.dtype != np.bool_ or int(mask.sum()) != 2051:
        raise BundleError("Feature mask tidak cocok dengan run final.")
    mask.setflags(write=False)

    models = []
    entries = manifest["models"]
    if len(entries) != len(labels):
        raise BundleError("Jumlah model tidak sesuai dengan daftar label.")
    for index, (label, entry) in enumerate(zip(labels, entries, strict=True)):
        if entry["label"] != label:
            raise BundleError("Urutan label dan file model tidak sesuai.")
        if "constant" in entry:
            if entry["constant"] not in (0.0, 1.0):
                raise BundleError("Skor konstan tidak valid.")
            models.append(float(entry["constant"]))
            continue
        relative = f"models/label_{index:03d}.ubj.gz"
        if entry["path"] != relative:
            raise BundleError("Nama file model tidak sesuai.")
        booster = Booster()
        booster.load_model(bytearray(gzip.decompress(verified(relative))))
        booster.set_param({"device": "cpu", "nthread": 1})
        if booster.num_features() != int(mask.sum()):
            raise BundleError("Dimensi model tidak cocok dengan feature mask.")
        models.append(booster)
    return Bundle(manifest["version"], labels, manifest["threshold"], mask, tuple(models))
