from typing import List
import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from lightning_classes import utils
from lightning.pytorch.loggers import Logger
import os 

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

@hydra.main(config_path="config/", config_name="config.yaml", version_base = "1.2")
def main(cfg: DictConfig):
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    # Lightning datamodule class would be found under language_modelling/data
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    # Lightning model class would be found under language_modelling/model
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks= callbacks)
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()