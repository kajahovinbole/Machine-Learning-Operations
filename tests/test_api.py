from unittest.mock import patch

from fastapi.testclient import TestClient

from src.clickbait_classifier.api import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "OK", "status-code": 200}


@patch("src.clickbait_classifier.api.tokenizer")
@patch("src.clickbait_classifier.api.model")
def test_predict_endpoint(mock_model, mock_tokenizer):
    with TestClient(app) as client:
        response = client.post("/predict?text=Dette er en test")
        assert response.status_code == 200
        assert "is_clickbait" in response.json()
