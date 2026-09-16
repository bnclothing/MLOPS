from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "bioactivity_mlp_scripted.pt"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"


def _resolve_path(environment_name: str, default: Path) -> Path:
    configured = os.getenv(environment_name)
    return Path(configured).expanduser().resolve() if configured else default


def _select_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE=cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError("MODEL_DEVICE must be one of: auto, cpu, cuda")


class BioactivityPredictor:
    """Loads the exported model once and performs thread-safe batched inference."""

    def __init__(
        self,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path or _resolve_path("MODEL_PATH", DEFAULT_MODEL_PATH)
        self.metadata_path = metadata_path or _resolve_path("MODEL_METADATA_PATH", DEFAULT_METADATA_PATH)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {self.metadata_path}")

        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        requested_device = device or os.getenv("MODEL_DEVICE", "auto")
        self.device = _select_device(requested_device)
        self.radius = int(self.metadata["input"]["radius"])
        self.n_bits = int(self.metadata["input"]["n_bits"])
        self.threshold = float(self.metadata["decision_threshold"])
        self.target_chembl_id = str(self.metadata["target"]["chembl_id"])
        self.target_name = str(self.metadata["target"]["name"])
        self.generator = AllChem.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        self.uncharger = rdMolStandardize.Uncharger()
        self.model = torch.jit.load(str(self.model_path), map_location=self.device).eval()
        self._inference_lock = threading.Lock()

        # Fail during startup if the artifact and metadata dimensions do not agree.
        with torch.inference_mode():
            smoke_input = torch.zeros((1, self.n_bits), dtype=torch.float32, device=self.device)
            smoke_output = self.model(smoke_input)
        if tuple(smoke_output.shape) != (1,):
            raise RuntimeError(f"Unexpected model output shape: {tuple(smoke_output.shape)}")

    def standardize_smiles(self, smiles: str) -> str | None:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = rdMolStandardize.Cleanup(mol)
            mol = rdMolStandardize.FragmentParent(mol)
            mol = self.uncharger.uncharge(mol)
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return None

    def _fingerprint(self, standardized_smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(standardized_smiles)
        if mol is None:
            raise ValueError("Cannot fingerprint an invalid standardized SMILES")
        array = np.zeros(self.n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(self.generator.GetFingerprint(mol), array)
        return array

    def predict_many(
        self,
        molecules: Iterable[tuple[str | None, str]],
        inference_batch_size: int = 1_024,
    ) -> list[dict]:
        molecule_list = list(molecules)
        results: list[dict] = []
        valid_positions: list[int] = []
        fingerprints: list[np.ndarray] = []

        for molecule_id, smiles in molecule_list:
            standardized = self.standardize_smiles(smiles)
            row = {
                "id": molecule_id,
                "smiles": smiles,
                "standardized_smiles": standardized,
                "valid": standardized is not None,
                "probability_active": None,
                "prediction": "invalid_smiles",
                "decision_threshold": self.threshold,
                "target_chembl_id": self.target_chembl_id,
                "target_name": self.target_name,
                "error": None if standardized is not None else "SMILES could not be parsed or standardized",
            }
            results.append(row)
            if standardized is not None:
                valid_positions.append(len(results) - 1)
                fingerprints.append(self._fingerprint(standardized))

        if not fingerprints:
            return results

        matrix = np.vstack(fingerprints)
        probability_chunks: list[np.ndarray] = []
        with self._inference_lock, torch.inference_mode():
            for start in range(0, len(matrix), inference_batch_size):
                features = torch.from_numpy(matrix[start : start + inference_batch_size]).to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                probability_chunks.append(torch.sigmoid(self.model(features)).cpu().numpy())
        probabilities = np.concatenate(probability_chunks)

        for result_position, probability in zip(valid_positions, probabilities, strict=True):
            score = float(probability)
            results[result_position]["probability_active"] = score
            results[result_position]["prediction"] = "active" if score >= self.threshold else "inactive"
        return results

    def predict_one(self, smiles: str) -> dict:
        return self.predict_many([(None, smiles)])[0]

