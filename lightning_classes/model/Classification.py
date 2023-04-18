from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from transformers import AutoModelForMaskedLM, AutoTokenizer

def calculate_accuracy(big_idx, targets):
    """

    Args:
        big_idx:
        targets:

    Returns:

    """
    n_correct = (big_idx == targets).sum().item()
    return n_correct

class HierarchicalSSOCClassifier(LightningModule):
    def __init__(self, model: None,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        net: None, 
        other_params: None) -> None:
        super().__init__()

        self.save_hyperparameters(logger=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model.tokenizer)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = MeanMetric()
        self.val_acc = MeanMetric()

        self.val_acc_best = MaxMetric()
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, batch_title_ids, batch_title_mask, batch_text_ids, batch_text_mask):
        return self.net(batch_title_ids, batch_title_mask, batch_text_ids, batch_text_mask)
    
    def on_train_start(self):
        self.val_loss.reset()

    def model_step(self, batch: Any):
        predictions = self.forward(batch['title_ids'], batch['title_mask'], batch['text_ids'], batch['text_mask'])
        weighted_loss = {'SSOC_1D': 20, 'SSOC_2D': 10, 'SSOC_3D': 5, 'SSOC_4D': 2, 'SSOC_5D': 1}
        loss = 0
        for ssoc_level, preds in predictions.items():
            # Extract the correct target for the SSOC level
            targets = batch[ssoc_level].to('cpu', dtype=torch.long)
            # Compute the loss function using the predictions and the targets
            level_loss = self.criterion(preds, targets)
            # Initialise the loss variable if this is the 1D level
            # Else add to the loss variable
            # Note the weights on each level
            loss += level_loss * weighted_loss[ssoc_level]

        top_probs, top_probs_idx = torch.max(preds.data, dim=1)
        acc = calculate_accuracy(top_probs_idx, targets)
        return loss, acc

    def training_step(self, batch: Any, batch_idx: int):
        loss, acc = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(acc)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, acc = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(acc)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, acc = self.model_step(batch)
        self.test_loss(loss)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer, 
                            num_warmup_steps=self.hparams.other_params.num_warmup_steps, 
                            num_training_steps=self.hparams.other_params.num_training_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
if __name__ == "__main__":
    _ = HierarchicalSSOCClassifier()