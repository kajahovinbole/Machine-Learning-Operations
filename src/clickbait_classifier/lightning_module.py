"""PyTorch Lightning module for clickbait classifier."""

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy

from clickbait_classifier.model import ClickbaitClassifier


class ClickbaitLightningModule(pl.LightningModule):
    """Lightning wrapper for ClickbaitClassifier."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = ClickbaitClassifier(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
        )

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)

        self.train_acc(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)

        self.val_acc(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
