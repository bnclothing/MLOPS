from fastapi.testclient import TestClient

from app.main import app


CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"


def test_root_health_and_model_info() -> None:
    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["documentation"] == "/docs"

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["model_loaded"] is True
        assert health.json()["target_chembl_id"] == "CHEMBL203"

        model_info = client.get("/model-info")
        assert model_info.status_code == 200
        assert model_info.json()["input"]["n_bits"] == 2048


def test_single_prediction() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"smiles": CAFFEINE})
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is True
        assert 0.0 <= body["probability_active"] <= 1.0
        assert body["prediction"] in {"active", "inactive"}
        assert body["target_chembl_id"] == "CHEMBL203"


def test_invalid_single_smiles_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"smiles": "not-a-smiles"})
        assert response.status_code == 422
        assert "could not be parsed" in response.json()["detail"]["message"]


def test_batch_prediction_preserves_ids_and_invalid_rows() -> None:
    payload = {
        "molecules": [
            {"id": "caffeine", "smiles": CAFFEINE},
            {"id": "gefitinib", "smiles": GEFITINIB},
            {"id": "bad", "smiles": "not-a-smiles"},
        ]
    }
    with TestClient(app) as client:
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert [item["id"] for item in body["predictions"]] == ["caffeine", "gefitinib", "bad"]
        assert body["summary"] == {
            "total": 3,
            "valid": 2,
            "invalid": 1,
            "predicted_active": sum(
                item["prediction"] == "active" for item in body["predictions"]
            ),
            "predicted_inactive": sum(
                item["prediction"] == "inactive" for item in body["predictions"]
            ),
        }


def test_request_validation() -> None:
    with TestClient(app) as client:
        assert client.post("/predict", json={"smiles": "   "}).status_code == 422
        assert client.post("/predict", json={"smiles": CAFFEINE, "unknown": 1}).status_code == 422
        assert client.post("/predict/batch", json={"molecules": []}).status_code == 422

