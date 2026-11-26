import pytest

from odor_app.artifacts import BUNDLE, load_bundle, manifest_digest
from odor_app.inference import Predictor


@pytest.fixture(scope="session")
def predictor():
    return Predictor(load_bundle(BUNDLE, manifest_digest()))
