import torch
import typer
from torch import nn
from transformers import AutoModel

app = typer.Typer()


class ClickbaitClassifier(nn.Module):
    """DistilBERT-based classifier for clickbait detection."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


@app.command()
def info() -> None:
    """Show model architecture and parameter count."""
    model = ClickbaitClassifier()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model: ClickbaitClassifier")
    print(f"Base: distilbert-base-uncased")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


@app.command()
def test() -> None:
    """Run a quick forward pass test."""
    model = ClickbaitClassifier()
    model.eval()

    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    print(f"Input shape: ({batch_size}, {seq_length})")
    print(f"Output shape: {logits.shape}")
    print("Forward pass successful!")


if __name__ == "__main__":
    app()
