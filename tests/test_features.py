import numpy as np
import pytest

from odor_app.features import Featurizer, InvalidSmiles


@pytest.mark.parametrize("smiles, code", [
    ("", "empty"), ("NO!", "parse"), ("CCO.O", "fragments"),
    ("CCO ethanol", "whitespace"), ("C" * 513, "too_long"), ("C*", "wildcard"),
])
def test_rejects_out_of_scope_smiles(smiles, code):
    with pytest.raises(InvalidSmiles) as error:
        Featurizer().transform(smiles)
    assert error.value.code == code


def test_ethanol_fingerprint_and_descriptors():
    canonical, features = Featurizer().transform(" OCC ")
    assert canonical == "CCO"
    assert features.shape == (2056,) and features.dtype == np.float32
    assert np.isin(features[:2048], (0, 1)).all() and features[:2048].sum() > 0
    np.testing.assert_allclose(features[2048:], [46.069, -0.0014, 1, 1, 20.23, 0, 3, 0], atol=2e-5)
