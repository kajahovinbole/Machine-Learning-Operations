# Clickbait Classifier

## Overall Goal

We want to build a clickbait detector, a model that can tell whether a headline is a genuine news or sensationalized garbage. The idea is to train a text classifier and deploy it as an API that can score headlines in real time.

The ML task itself isn't groundbreaking, but that's kind of the point. We want to spend our time on the MLOps side: setting up reproducible training, experiment tracking, CI/CD, containerization, and cloud deployment.

## Framework

We'll use PyTorch as our deep learning framework and HuggingFace Transformers for pretrained models and tokenizers.

## Data

We're using the [Clickbait Dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset) from Kaggle. It has around 32,000 headlines labeled as clickbait or not. Small enough to iterate quickly, clean enough that we won't spend days on preprocessing or training.

## Models

We haven't fully decided on our approach yet. Options we're considering:

- Fine-tuning a pre-trained model using HuggingFace Transformers, something like DistilBERT, BERT, or RoBERTa. This is probably the most practical path.
- Training a simpler model from scratch to compare and understand the difference.
- Both, if time allows. Comparing fine-tuned transformers against a baseline we build ourselves.

Model candidates include `distilbert-base-uncased` (fast, good enough), `bert-base-uncased` (standard choice), or potentially something smaller if inference speed matters for deployment.

We'll track experiments in W&B and let the results guide our final choice. The priority is getting the pipeline working end-to-end, not squeezing out the last percentage of accuracy.


# How to run

## Docker

This project uses Docker and Docker Compose to run the training and API
in a reproducible environment.


### Prerequisites
- Docker Desktop installed and running

---

### Run API
Builds the image (if needed) and starts the API service:

```bash
uv run invoke docker-up api
```

The API will be available at: http://localhost:8000/docs
Stop with Ctrl+C.

## Run training
Builds the image (if needed), runs the training container once, and exits:

```bash
docker compose run --rm --build train
```

## Run all services
Builds images (if needed) and starts both the API and training services:

```bash
uv run invoke docker-up
```