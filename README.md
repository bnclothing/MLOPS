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

The repository includes the trained deployment artifacts, so API users do **not** need to
run the notebook or retrain the model:

- `artifacts/bioactivity_mlp_scripted.pt` — model used by FastAPI;
- `artifacts/model_metadata.json` — preprocessing parameters, threshold, and target details;
- `artifacts/bioactivity_mlp.onnx` — optional interoperable export;
- `artifacts/bioactivity_mlp_state_dict.pt` — optional PyTorch weights for future training work.

Their SHA-256 checksums are recorded in `artifacts/SHA256SUMS.txt`.

### Fresh-clone API setup (no training)

For a friend with no NVIDIA GPU:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
.\start_api.ps1 -Device cpu
```

For a friend with a compatible NVIDIA GPU, replace `requirements-cpu.txt` with
`requirements-cuda.txt` and start with `-Device cuda`. On Linux or macOS, activate the
environment and run `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`.

## Utiliser le modèle exporté

```powershell
.\.venv\Scripts\python.exe predict.py --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
.\.venv\Scripts\python.exe predict.py --input molecules.csv --smiles-column smiles --output predictions.csv
```

## FastAPI

Start the API on the local machine and force CUDA:

```powershell
.\start_api.ps1 -Device cuda
```

The service is then available at:

- API: `http://127.0.0.1:8000`
- interactive Swagger documentation: `http://127.0.0.1:8000/docs`
- health check: `http://127.0.0.1:8000/health`

Single prediction:

```powershell
$body = @{ smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" -Body $body
```

Batch prediction:

```powershell
$body = @{
  molecules = @(
    @{ id = "caffeine"; smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
    @{ id = "gefitinib"; smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1" }
  )
} | ConvertTo-Json -Depth 4

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict/batch" `
  -ContentType "application/json" -Body $body
```

Environment variables are documented in `.env.example`. By default, `MODEL_DEVICE=auto`
uses CUDA when it is available and falls back to CPU otherwise.

Run the API tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

> Le modèle est destiné à la priorisation *in silico*. Il ne remplace pas une validation expérimentale et son domaine d'applicabilité doit être vérifié avant toute décision scientifique.
