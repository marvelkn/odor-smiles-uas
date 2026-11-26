"""The v2 training representation for a single, complete molecule."""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

MAX_SMILES_LENGTH = 512
FEATURE_COUNT = 2056
DESCRIPTORS = (
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA",
    "NumRotatableBonds", "HeavyAtomCount", "RingCount",
)

# RDKit's parser diagnostics can include the full string submitted by the user.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


class InvalidSmiles(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Featurizer:
    def __init__(self):
        self._morgan = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=False
        )

    def transform(self, value: str) -> tuple[str, np.ndarray]:
        value = value.strip()
        if not value:
            raise InvalidSmiles("empty", "Masukkan SMILES atau pilih contoh molekul.")
        if len(value) > MAX_SMILES_LENGTH:
            raise InvalidSmiles("too_long", f"SMILES maksimal {MAX_SMILES_LENGTH} karakter.")
        if any(character.isspace() for character in value):
            raise InvalidSmiles("whitespace", "Masukkan satu SMILES tanpa nama atau spasi di tengah.")

        options = Chem.SmilesParserParams()
        options.parseName = False
        options.allowCXSMILES = False
        molecule = Chem.MolFromSmiles(value, options)
        if molecule is None or not molecule.GetNumAtoms():
            raise InvalidSmiles("parse", "SMILES tidak valid. Periksa simbol atom, ikatan, dan tanda kurung.")
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise InvalidSmiles("fragments", "Masukkan satu molekul. Campuran dan struktur multikomponen belum didukung.")
        if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
            raise InvalidSmiles("wildcard", "Ganti atom wildcard (*) dengan struktur molekul yang lengkap.")

        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        # Training featurized the reparsed canonical string, so match that order.
        molecule = Chem.MolFromSmiles(canonical, options)
        if molecule is None:
            raise InvalidSmiles("canonical", "Struktur ini tidak dapat diproses.")
        features = np.empty(FEATURE_COUNT, dtype=np.float32)
        features[:2048] = self._morgan.GetFingerprintAsNumPy(molecule)
        features[2048:] = [getattr(Descriptors, name)(molecule) for name in DESCRIPTORS]
        if not np.isfinite(features).all():
            raise InvalidSmiles("nonfinite", "Deskriptor untuk struktur ini tidak dapat dihitung.")
        return canonical, features
