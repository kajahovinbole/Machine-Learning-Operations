from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from clickbait_classifier.data import load_data
from clickbait_classifier.model import ClickbaitClassifier


# Endre denne delen i _load_config:
def _load_config(config_path: Optional[Path]) -> OmegaConf:
    """Load configuration from file using Hydra."""
    if config_path is None:
        config_path = Path("configs/config.yaml")

    config_path = Path(config_path).resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem

    # Vi bruker initialize_config_dir direkte uten å importere version_base først
    # Dette er mer robust når miljøet er litt rotete
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained clickbait model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt (state_dict).")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml used for training.")
    parser.add_argument("--processed-path", type=str, default=None, help="Path to processed data (overrides config).")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Which split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config).")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--out", type=str, default="reports/evaluation.json", help="Where to save results (json).")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
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


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    args = parse_args()

    cfg = _load_config(Path(args.config)) if args.config else _load_config(None)

    # overrides
    if args.processed_path is not None:
        cfg.data.processed_path = str(args.processed_path)
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size

    device = _resolve_device(args.device)
    logger.info(f"Using device: {device}")

    # data
    processed_path = Path(cfg.data.processed_path)
    train_set, val_set, test_set = load_data(processed_path)
    dataset = val_set if args.split == "val" else test_set

    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)
    logger.info(f"Evaluating split='{args.split}' with {len(dataset)} samples")

    # model
    model = ClickbaitClassifier(
        model_name=cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        dropout=cfg.model.dropout,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    results = evaluate(model, loader, device)
    results.update(
        {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "device": str(device),
            "batch_size": cfg.training.batch_size,
            "model_name": cfg.model.model_name,
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved evaluation to {out_path}")


if __name__ == "__main__":
    main()

