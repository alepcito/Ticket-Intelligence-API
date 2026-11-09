import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from training.train import ROOT, train


@pytest.fixture(scope="session")
def artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("model")
    train(ROOT / "data/tickets.csv", directory)
    return directory / "model.joblib"


@pytest.fixture
def client(artifact):
    with TestClient(create_app(artifact)) as session:
        yield session


def test_health_and_metadata(client):
    assert client.get("/health").json()["status"] == "ok"
    metadata = client.get("/model").json()["metadata"]
    assert metadata["train_samples"] == 60
    assert metadata["test_samples"] == 20
    assert len(metadata["categories"]) == 5


def test_prediction_contract(client):
    response = client.post("/predict", json={"text": "Necesito una factura del pago"})
    assert response.status_code == 200
    result = response.json()
    assert result["category"] == "facturacion"
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["confidence"] == max(result["probabilities"].values())
    assert result["needs_review"] == (result["confidence"] < 0.6)
    assert "X-Process-Time-Ms" in response.headers


@pytest.mark.parametrize("payload", [{}, {"text": "     "}, {"text": "x" * 5001}, {"text": 123}])
def test_invalid_tickets(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_matches_individual_predictions(client):
    tickets = [{"text": "Quiero cancelar mi cuenta"}, {"text": "Necesito recuperar mi contraseña"}]
    result = client.post("/predict/batch", json={"tickets": tickets})
    assert result.status_code == 200
    assert result.json() == [client.post("/predict", json=ticket).json() for ticket in tickets]


@pytest.mark.parametrize("tickets", [[], [{"text": "Ticket de prueba"}] * 101])
def test_batch_limits(client, tickets):
    assert client.post("/predict/batch", json={"tickets": tickets}).status_code == 422


def test_review_threshold(artifact):
    with TestClient(create_app(artifact, threshold=1.0)) as client:
        assert client.post("/predict", json={"text": "Un mensaje ambiguo"}).json()["needs_review"]


def test_missing_model(tmp_path):
    with pytest.raises(RuntimeError, match="Model missing"):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass


def test_reproducible_training(tmp_path):
    first = train(ROOT / "data/tickets.csv", tmp_path / "first")
    second = train(ROOT / "data/tickets.csv", tmp_path / "second")
    assert first == second
    assert first["macro_f1"] > first["baseline_macro_f1"]
