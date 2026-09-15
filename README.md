# Criblage virtuel de molécules bioactives — ChEMBL / EGFR

Ce projet entraîne un classifieur de bioactivité à partir des mesures `IC50` de la cible humaine EGFR (`CHEMBL203`) publiées dans ChEMBL.

Le pipeline complet est dans `virtual_screening_chembl.ipynb` :

1. téléchargement paginé et mise en cache des activités ChEMBL ;
2. standardisation RDKit et déduplication des molécules ;
3. labels actifs/inactifs à partir de `pChEMBL` (zone ambiguë exclue) ;
4. fingerprints Morgan (2048 bits) ;
5. séparation train/validation/test par scaffold de Bemis–Murcko ;
6. réseau MLP PyTorch pondéré, early stopping et CUDA ;
7. métriques ROC-AUC, PR-AUC, MCC, matrice de confusion et choix du seuil ;
8. export du modèle (`TorchScript`, `state_dict`, `ONNX`) et des métadonnées ;
9. fonction de scoring de nouvelles molécules SMILES.

## Démarrage

```powershell
.\.venv\Scripts\Activate.ps1
jupyter lab virtual_screening_chembl.ipynb
```

Pour recréer l'environnement avec Python 3.12 :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Les artefacts entraînés sont écrits dans `artifacts/`. Les données téléchargées sont mises en cache dans `data/raw/`.

## Utiliser le modèle exporté

```powershell
.\.venv\Scripts\python.exe predict.py --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
.\.venv\Scripts\python.exe predict.py --input molecules.csv --smiles-column smiles --output predictions.csv
```

> Le modèle est destiné à la priorisation *in silico*. Il ne remplace pas une validation expérimentale et son domaine d'applicabilité doit être vérifié avant toute décision scientifique.
