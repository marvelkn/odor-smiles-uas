"""Shared presentation helpers and per-process loading."""

import json
import logging
from pathlib import Path

import streamlit as st

from .artifacts import BUNDLE, load_bundle, manifest_digest
from .inference import Predictor

DATA = Path(__file__).resolve().parents[1] / "data"
logger = logging.getLogger("odor_app")


@st.cache_resource(max_entries=1, show_spinner=False)
def _predictor(manifest_sha: str) -> Predictor:
    return Predictor(load_bundle(BUNDLE, manifest_sha))


def get_predictor() -> Predictor:
    with st.spinner("Menyiapkan model…"):
        return _predictor(manifest_digest())


@st.cache_data(max_entries=2, show_spinner=False)
def _decode_public_data(contents: bytes):
    return json.loads(contents)


def public_data(name: str):
    # Streamlit may reload app code without clearing the previous data cache.
    return _decode_public_data((DATA / name).read_bytes())


def log_event(event: str, status: str, elapsed: float, row_count: int, version: str = "unknown") -> None:
    # Exceptions and submitted values are deliberately omitted from cloud logs.
    logger.warning("event=%s status=%s duration_ms=%d rows=%d model=%s",
                   event, status, round(elapsed * 1000), row_count, version)
