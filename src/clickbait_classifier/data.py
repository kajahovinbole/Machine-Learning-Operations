from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from torch.utils.data import Dataset, TensorDataset
from transformers import AutoTokenizer

app = typer.Typer()


class ClickbaitDataset(Dataset):
    """Dataset for clickbait classification (raw text)."""

    def __init__(self, data_path: Path) -> None:
        self.data = pd.read_csv(data_path)
        self.headlines = self.data["headline"].tolist()
        self.labels = torch.tensor(self.data["clickbait"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        return self.headlines[index], self.labels[index]


def load_data(
    processed_path: Path = Path("data/processed"),
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Load preprocessed train, val, and test sets."""
    train_data = torch.load(processed_path / "train.pt")
    val_data = torch.load(processed_path / "val.pt")
    test_data = torch.load(processed_path / "test.pt")

    train_set = TensorDataset(
        train_data["input_ids"],
        train_data["attention_mask"],
        train_data["labels"],
    )
    val_set = TensorDataset(
        val_data["input_ids"],
        val_data["attention_mask"],
        val_data["labels"],
    )
    test_set = TensorDataset(
        test_data["input_ids"],
        test_data["attention_mask"],
        test_data["labels"],
    )

    return train_set, val_set, test_set


@app.command()
def preprocess(
    raw_path: Annotated[Path, typer.Option(help="Path to raw CSV")] = Path("data/raw/clickbait_data.csv"),
    output_path: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("data/processed"),
    model_name: Annotated[str, typer.Option(help="Tokenizer model name")] = "distilbert-base-uncased",
    max_length: Annotated[int, typer.Option(help="Max sequence length")] = 128,
    train_split: Annotated[float, typer.Option(help="Train split ratio")] = 0.7,
    val_split: Annotated[float, typer.Option(help="Validation split ratio")] = 0.15,
) -> None:
    """Tokenize raw data and save train/val/test splits as tensors."""
    print(f"Loading data from {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} samples")
    print(f"Clickbait: {df['clickbait'].sum()}, Non-clickbait: {(df['clickbait'] == 0).sum()}")

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split indices
    n = len(df)
    train_end = int(train_split * n)
    val_end = int((train_split + val_split) * n)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Tokenize
    print(f"Tokenizing with {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        encodings = tokenizer(
            split_df["headline"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        data = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(split_df["clickbait"].values, dtype=torch.long),
        }

        torch.save(data, output_path / f"{split_name}.pt")
        print(f"Saved {split_name}.pt with shape {encodings['input_ids'].shape}")


if __name__ == "__main__":
    app()
