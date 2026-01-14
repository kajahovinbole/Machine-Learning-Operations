from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import TensorDataset

import clickbait_classifier.lightning_module as lightning_module
import clickbait_classifier.train as train_module


class DummyModel(nn.Module):
    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": hidden_size})()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = input_ids.float().mean(dim=1, keepdim=True)
        return self.classifier(x)

def _dummy_cfg(tmp_path: Path):
    return SimpleNamespace(
        data=SimpleNamespace(processed_path=str(tmp_path / "data" / "processed")),
        model=SimpleNamespace(model_name="dummy", num_labels=2, dropout=0.0),
        training=SimpleNamespace(
            seed=123,
            device="cpu",
            epochs=1,
            batch_size=4,
            lr=1e-3,
            shuffle=False,
            optimizer=SimpleNamespace(),
            optimizer=SimpleNamespace(),
            loss=SimpleNamespace(),
        ),
        paths=SimpleNamespace(model_output=str(tmp_path / "models" / "clickbait_model.pt")),
    )


def _dummy_datasets():
    num_samples, seq_len = 8, 16
    input_ids = torch.randint(0, 1000, (num_samples, seq_len), dtype=torch.long)
    attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)
    labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)
    ds = TensorDataset(input_ids, attention_mask, labels)
    return ds, ds, ds


def test_train_runs_with_lightning(monkeypatch, tmp_path, patch_transformer):
    cfg = _dummy_cfg(tmp_path)

    # gå til tmp_path så relative paths ("models/...") havner der
    monkeypatch.chdir(tmp_path)

    saved = {"config_path": None}

    monkeypatch.setattr(train_module, "_load_config", lambda _config_path: cfg)
    monkeypatch.setattr(train_module, "load_data", lambda _processed_path: _dummy_datasets())

    def fake_save_config(_cfg, path):
        saved["config_path"] = Path(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("# dummy config")

    monkeypatch.setattr(train_module, "save_config", fake_save_config)
    monkeypatch.setattr(train_module, "api_key", None)

    train_module.train()

    assert saved["config_path"] is not None

    run_dirs = list((tmp_path / "models").glob("2*"))
    assert len(run_dirs) > 0, "No model directory created"

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    ckpt_files = list(latest_run.glob("*.ckpt"))
    assert len(ckpt_files) > 0, f"No checkpoint saved in {latest_run}"


