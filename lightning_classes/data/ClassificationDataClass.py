from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from lightning_classes.data.processing import tokenize_function, group_texts, split_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from ssoc_autocoder.model_training import import_ssoc_idx_encoding, encode_dataset, SSOC_Dataset
import pandas as pd

class AdsDataModule(LightningDataModule):
    def __init__(self, data_dir: None, test_dir: None,
        batch_size: 128,
        num_workers: 0,
        persistant_workers: False,
        pin_memory: False,
        chunk_size: 128,
        seed: 1,
        normal_masking_probability: 0.2,
        train_size: 8,
        fraction: 0.2, 
        tokenizer: None,
        ssoc_idx_encoding_filepath: None
        ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # set of functions to carry out data transformations
        # e.g. tokenization and token concatenation
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None 

        self.columns = {'SSOC':'SSOC', 'job_title':'job_title', 'job_description':'job_description'}
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def prepare_data(self) -> None:
        encoding = import_ssoc_idx_encoding(self.hparams.ssoc_idx_encoding_filepath)
        train_data = pd.read_csv(self.hparams.data_dir)
        self.encode_train = encode_dataset(train_data, encoding, colnames=self.columns)
        test_data = pd.read_csv(self.hparams.test_dir)
        self.encode_test = encode_dataset(test_data, encoding, colnames=self.columns)

    def setup(self, stage: Optional[str] = None):
        self.data_train = self.encode_train
        self.data_val = self.encode_test

    def train_dataloader(self):
        return DataLoader(
            dataset=SSOC_Dataset(self.data_train, self.tokenizer, 512, self.columns, 'hierarchical'),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistant_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=SSOC_Dataset(self.data_val, self.tokenizer, 512, self.columns, 'hierarchical'),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistant_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistant_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass
        

if __name__ == "__main__":
    _ = AdsDataModule()