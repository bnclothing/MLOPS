"""Inférence autonome avec le modèle EGFR exporté.

Exemples :
  python predict.py --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
  python predict.py --input molecules.csv --smiles-column smiles --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "artifacts" / "bioactivity_mlp_scripted.pt"
DEFAULT_METADATA = ROOT / "artifacts" / "model_metadata.json"


def standardize_smiles(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def predict(
    smiles: list[str], model_path: Path, metadata_path: Path, batch_size: int = 1024
) -> pd.DataFrame:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    radius = int(metadata["input"]["radius"])
    n_bits = int(metadata["input"]["n_bits"])
    threshold = float(metadata["decision_threshold"])
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    model = torch.jit.load(str(model_path), map_location="cpu").eval()

    standardized: list[str | None] = []
    valid_indices: list[int] = []
    fingerprints: list[np.ndarray] = []
    for index, value in enumerate(smiles):
        canonical = standardize_smiles(str(value))
        standardized.append(canonical)
        if canonical is None:
            continue
        mol = Chem.MolFromSmiles(canonical)
        array = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), array)
        valid_indices.append(index)
        fingerprints.append(array)

    probabilities = np.full(len(smiles), np.nan, dtype=float)
    if fingerprints:
        matrix = np.vstack(fingerprints)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(matrix), batch_size):
                logits = model(torch.from_numpy(matrix[start : start + batch_size]))
                chunks.append(torch.sigmoid(logits).numpy())
        probabilities[np.asarray(valid_indices)] = np.concatenate(chunks)

    frame = pd.DataFrame(
        {
            "smiles": smiles,
            "standardized_smiles": standardized,
            "probability_active": probabilities,
        }
    )
    frame["prediction"] = np.where(
        frame["probability_active"].isna(),
        "invalid_smiles",
        np.where(frame["probability_active"] >= threshold, "active", "inactive"),
    )
    frame["decision_threshold"] = threshold
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score des molécules pour la bioactivité EGFR.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", nargs="+", help="Un ou plusieurs SMILES.")
    source.add_argument("--input", type=Path, help="Fichier CSV contenant les SMILES.")
    parser.add_argument("--smiles-column", default="smiles", help="Colonne du CSV (défaut: smiles).")
    parser.add_argument("--output", type=Path, help="CSV de sortie. Sinon, affiche le résultat.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        source_frame = pd.read_csv(args.input)
        if args.smiles_column not in source_frame:
            raise KeyError(f"Colonne absente: {args.smiles_column}")
        result = predict(source_frame[args.smiles_column].astype(str).tolist(), args.model, args.metadata)
        result = pd.concat([source_frame.reset_index(drop=True), result.drop(columns="smiles")], axis=1)
    else:
        result = predict(args.smiles, args.model, args.metadata)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Prédictions écrites dans {args.output.resolve()}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
