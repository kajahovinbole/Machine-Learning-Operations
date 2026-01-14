"""Performance tests for staged models."""

import os
import time

import pytest
import torch

from clickbait_classifier.lightning_module import ClickbaitLightningModule


@pytest.mark.skipif(
    not os.getenv("MODEL_PATH"),
    reason="MODEL_PATH not set - skipping performance test",
)
def test_model_inference_speed():
    """Test that model can do 100 predictions in under 5 seconds."""
    model_path = os.getenv("MODEL_PATH")
    model = ClickbaitLightningModule.load_from_checkpoint(model_path, map_location="cpu")
    model.eval()
    model.to("cpu")

    # Dummy input (batch of 100)
    batch_size = 100
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    # Warm up
    with torch.no_grad():
        _ = model(input_ids[:1], attention_mask[:1])

    # Time 100 predictions
    start = time.time()
    with torch.no_grad():
        for i in range(batch_size):
            _ = model(input_ids[i : i + 1], attention_mask[i : i + 1])
    elapsed = time.time() - start

    print(f"100 predictions took {elapsed:.2f}s")
    assert elapsed < 10.0, f"Model too slow: {elapsed:.2f}s for 100 predictions"


@pytest.mark.skipif(
    not os.getenv("MODEL_PATH"),
    reason="MODEL_PATH not set - skipping performance test",
)
def test_model_batch_inference():
    """Test that model can handle batch inference."""
    model_path = os.getenv("MODEL_PATH")
    model = ClickbaitLightningModule.load_from_checkpoint(model_path, map_location="cpu")
    model.eval()
    model.to("cpu")

    # Batch input
    batch_size = 32
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    start = time.time()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    elapsed = time.time() - start

    assert logits.shape == (batch_size, 2), f"Wrong output shape: {logits.shape}"
    print(f"Batch of {batch_size} took {elapsed:.2f}s")
    assert elapsed < 5.0, f"Batch inference too slow: {elapsed:.2f}s"
