"""Compute scores in the original label order before ranking."""

from dataclasses import dataclass

import numpy as np

from .artifacts import Bundle
from .features import Featurizer


@dataclass(frozen=True)
class Prediction:
    smiles: str
    scores: tuple[tuple[str, float], ...]
    threshold: float
    model_version: str

    @property
    def passing(self) -> tuple[tuple[str, float], ...]:
        return tuple((label, score) for label, score in self.scores if score >= self.threshold)


class Predictor:
    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        self.features = Featurizer()

    def score_rows(self, features: np.ndarray) -> np.ndarray:
        masked = np.ascontiguousarray(features[:, self.bundle.mask], dtype=np.float32)
        values = np.empty((len(masked), len(self.bundle.labels)), dtype=np.float32)
        for index, model in enumerate(self.bundle.models):
            values[:, index] = model if isinstance(model, float) else model.inplace_predict(masked)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise RuntimeError("Model menghasilkan skor yang tidak valid.")
        return values

    def predict_many(self, rows: list[tuple[str, np.ndarray]]) -> list[Prediction]:
        if not rows:
            return []
        values = self.score_rows(np.stack([features for _, features in rows]))
        results = []
        for (canonical, _), scores in zip(rows, values, strict=True):
            pairs = tuple(zip(self.bundle.labels, map(float, scores), strict=True))
            ranked = tuple(sorted(pairs, key=lambda pair: -pair[1]))
            results.append(Prediction(canonical, ranked, self.bundle.threshold, self.bundle.version))
        return results

    def predict(self, value: str) -> Prediction:
        return self.predict_many([self.features.transform(value)])[0]
