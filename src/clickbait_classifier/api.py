from http import HTTPStatus
from fastapi import FastAPI
import torch
from transformers import AutoTokenizer
from clickbait_classifier.model import ClickbaitClassifier # Importerer din klasse


app = FastAPI()

# Vi definerer globale variabler som lastes ved oppstart
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    # Bruk samme modellnavn som i din ClickbaitClassifier
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialiser din modell og last vektene
    model = ClickbaitClassifier(model_name=model_name)
    # Husk å bytte ut denne stien med din faktiske lagrede modell-fil
    state_dict = torch.load("models/clickbait_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

@app.post("/predict")
async def predict(text: str):
    # 1. Tokenisering (tekst -> tall)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # 2. Inference (tall -> prediksjon)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        prediction = torch.argmax(logits, dim=1).item()
    
    return {"text": text, "is_clickbait": bool(prediction)}

@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response