"""Evaluate a trained clickbait model."""

import json
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from clickbait_classifier.data import load_data
from clickbait_classifier.lightning_module import ClickbaitLightningModule

app = typer.Typer()


def _load_config(config_path: Optional[Path]) -> OmegaConf:
    """Load configuration from file using Hydra."""
    if config_path is None:
        config_path = Path("configs/config.yaml")

    config_path = Path(config_path).resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


@torch.no_grad()
def run_evaluation(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc, "n_samples": total}


@app.command()
def evaluate(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-c", help="Path to model .ckpt file")],
    config: Annotated[Optional[Path], typer.Option("--config", help="Path to config.yaml")] = None,
    processed_path: Annotated[Optional[Path], typer.Option("--processed-path", help="Path to processed data")] = None,
    split: Annotated[str, typer.Option("--split", "-s", help="Which split to evaluate")] = "test",
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", "-b", help="Batch size")] = None,
    device: Annotated[str, typer.Option("--device", "-d", help="Device: auto, cpu, cuda, mps")] = "auto",
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to save results")] = None,
) -> None:
    """Evaluate a trained clickbait classifier."""
    cfg = _load_config(config)

    # overrides
    if processed_path is not None:
        cfg.data.processed_path = str(processed_path)
    if batch_size is not None:
        cfg.training.batch_size = batch_size

    device_resolved = _resolve_device(device)
    logger.info(f"Using device: {device_resolved}")

    # data
    processed_path = Path(cfg.data.processed_path)
    train_set, val_set, test_set = load_data(processed_path)
    dataset = val_set if split == "val" else test_set

    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)
    logger.info(f"Evaluating split='{split}' with {len(dataset)} samples")

    # Load Lightning model
    logger.info(f"Loading checkpoint: {checkpoint}")
    model = ClickbaitLightningModule.load_from_checkpoint(checkpoint)

    # Default output path is next to checkpoint
    if output is None:
        output = checkpoint.parent / "evaluation.json"

    results = run_evaluation(model, loader, device_resolved)
    results.update(
        {
            "split": split,
            "checkpoint": str(checkpoint),
            "device": str(device_resolved),
            "batch_size": cfg.training.batch_size,
            "model_name": model.hparams.model_name,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved evaluation to {output}")


if __name__ == "__main__":
    app()
