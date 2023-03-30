from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from transformers import AutoModelForMaskedLM, AutoTokenizer

class MLM(LightningModule):
    def __init__(self, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model: None,
        other_params: None) -> None:
        super().__init__()

        self.save_hyperparameters(logger=True)
        self.lm = AutoModelForMaskedLM.from_pretrained(self.hparams.model.filepath)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model.tokenizer)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
    
    def forward(self, batch):
        return self.lm(**batch)
    
    def on_train_start(self):
        self.val_loss.reset()

    def model_step(self, batch: Any):
        output = self.forward(batch)
        loss = output.loss
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        self.lm.train()
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        self.lm.eval()
        loss = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        self.lm.eval()
        loss = self.model_step(batch)
        self.test_loss(loss)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.lm.parameters())
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
    _ = MLM()