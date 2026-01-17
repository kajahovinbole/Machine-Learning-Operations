from unittest.mock import patch

from fastapi.testclient import TestClient

from src.clickbait_classifier.api import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "OK", "status-code": 200}


@patch("src.clickbait_classifier.api.os.path.getmtime")  # Fikser krasjen i loggen din
@patch("src.clickbait_classifier.api.glob.glob")
@patch("src.clickbait_classifier.api.ClickbaitClassifier")
@patch("src.clickbait_classifier.api.AutoTokenizer")
def test_predict_endpoint(mock_tokenizer, mock_classifier, mock_glob, mock_mtime):
    # 1. Vi returnerer et liksom-tidspunkt for filen
    mock_mtime.return_value = 123456789.0

    # 2. Vi later som glob finner en fil
    mock_glob.return_value = ["models/fake_model.ckpt"]

    # 3. Vi later som modellen og tokenizeren lastes fint
    mock_classifier.return_value = lambda x: x

    with TestClient(app) as client:
        response = client.post("/predict?text=Dette er en test")
        assert response.status_code == 200
        assert "is_clickbait" in response.json()
