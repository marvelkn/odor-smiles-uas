"""Bounded CSV parsing and spreadsheet-friendly results."""

from collections.abc import Iterable
import csv
from dataclasses import dataclass
from io import StringIO

from .features import InvalidSmiles
from .inference import Prediction, Predictor

MAX_BATCH_BYTES = 1024 * 1024
MAX_BATCH_ROWS = 100


def read_smiles_csv(raw: bytes) -> list[str]:
    if not raw or len(raw) > MAX_BATCH_BYTES:
        raise ValueError("Unggah CSV yang tidak kosong dan berukuran maksimal 1 MB.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Simpan CSV dengan encoding UTF-8.") from None
    if "\0" in text:
        raise ValueError("Berkas harus berupa CSV teks.")
    try:
        rows = csv.reader(StringIO(text, newline=""), strict=True)
        header = [item.strip().lower() for item in next(rows, [])]
        if header != ["smiles"]:
            raise ValueError("Gunakan satu kolom bernama smiles. Unduh template jika diperlukan.")
        values = []
        for row in rows:
            if not row:
                continue
            if len(row) != 1:
                raise ValueError("Setiap baris harus berisi satu kolom SMILES.")
            values.append(row[0])
            if len(values) > MAX_BATCH_ROWS:
                raise ValueError(f"CSV maksimal {MAX_BATCH_ROWS} baris data.")
    except csv.Error:
        raise ValueError("Format CSV rusak. Gunakan template UTF-8 dan coba lagi.") from None
    if not values:
        raise ValueError("CSV belum berisi baris data.")
    return values


@dataclass(frozen=True)
class BatchResult:
    row: int
    prediction: Prediction | None = None
    message: str = ""


def predict_csv(predictor: Predictor, values: list[str]) -> list[BatchResult]:
    valid_rows, features = [], []
    results = {}
    for index, value in enumerate(values, start=1):
        try:
            features.append(predictor.features.transform(value))
        except InvalidSmiles as error:
            results[index] = BatchResult(index, message=str(error))
        else:
            valid_rows.append(index)
    for index, prediction in zip(valid_rows, predictor.predict_many(features), strict=True):
        results[index] = BatchResult(index, prediction)
    return [results[index] for index in range(1, len(values) + 1)]


def make_csv(columns: Iterable[str], rows: Iterable[Iterable[object]]) -> bytes:
    def spreadsheet_safe(value: object) -> object:
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
            return "'" + value
        return value

    out = StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(columns)
    writer.writerows([spreadsheet_safe(value) for value in row] for row in rows)
    return out.getvalue().encode("utf-8-sig")


def single_csv(prediction: Prediction) -> bytes:
    return make_csv(
        ("smiles_kanonik", "label", "skor_model", "lolos_ambang", "ambang", "versi_model"),
        ((prediction.smiles, label, score, score >= prediction.threshold,
          prediction.threshold, prediction.model_version) for label, score in prediction.scores),
    )


def batch_csv(results: list[BatchResult], labels: tuple[str, ...], *, names: list[str] | None = None) -> bytes:
    def as_row(item: BatchResult):
        identity = [item.row] if names is None else [item.row, names[item.row - 1]]
        pred = item.prediction
        if pred is None:
            return [*identity, "invalid", item.message, "", "", "", "", *([""] * len(labels))]
        by_label = dict(pred.scores)
        return [*identity, "ok", "", pred.smiles, pred.model_version, pred.threshold,
                len(pred.passing), *(by_label[label] for label in labels)]

    identity_columns = ("baris_data",) if names is None else ("baris_data", "molekul")
    return make_csv((*identity_columns, "status", "pesan", "smiles_kanonik", "versi_model", "ambang",
                     "jumlah_lolos", *(f"skor_{label}" for label in labels)), map(as_row, results))
