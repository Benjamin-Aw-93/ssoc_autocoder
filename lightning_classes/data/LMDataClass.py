from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from lightning_classes.data.processing import tokenize_function, group_texts, split_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

class JobAdDataModule(LightningDataModule):
    def __init__(self, data_dir: None,
        batch_size: 128,
        num_workers: 0,
        persistant_workers: False,
        pin_memory: False,
        chunk_size: 128,
        seed: 1,
        normal_masking_probability: 0.2,
        train_size: 8,
        fraction: 0.2, 
        tokenizer: None
        ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # set of functions to carry out data transformations
        # e.g. tokenization and token concatenation
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None 
    
    def prepare_data(self) -> None:
        dataset = load_dataset('text', data_files = self.hparams.data_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)
        tokenized_dataset = dataset.map(lambda x: tokenize_function(dataset=x, tokenizer=tokenizer), batched = True, remove_columns=['text'])
        grouped_tokenized_datasets = tokenized_dataset.map(lambda x: group_texts(tokenized_dataset=x, chunk_size = self.hparams.chunk_size), batched=True)
        self.dataset = split_dataset(self.hparams.train_size, self.hparams.fraction, grouped_tokenized_datasets, self.hparams.seed)
        self.dataset = self.dataset.remove_columns(["word_ids"])
        self.data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm_probability=self.hparams.normal_masking_probability)

    def setup(self, stage: Optional[str] = None):
        self.data_train = self.dataset["train"]
        self.data_val = self.dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistant_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.data_collator
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistant_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.data_collator
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
    _ = JobAdDataModule()