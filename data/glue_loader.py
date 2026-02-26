# GLUE Data Loader — SST-2, MNLI, RTE
# Tái sử dụng từ bimamba_project, tương thích với TransMamba-Cls

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class GLUEDataModule:
    """
    Unified GLUE data loader cho SST-2, MNLI, RTE.
    
    Tasks:
    - SST-2: Sentiment (2 classes) - "sentence" → label
    - MNLI:  NLI (3 classes) - "premise" + "hypothesis" → label  
    - RTE:   Entailment (2 classes) - "sentence1" + "sentence2" → label
    """
    
    TASK_CONFIG = {
        "sst2": {
            "glue_name": "sst2",
            "text_fields": ["sentence"],
            "num_labels": 2,
            "label_names": ["negative", "positive"],
        },
        "mnli": {
            "glue_name": "mnli",
            "text_fields": ["premise", "hypothesis"],
            "num_labels": 3,
            "label_names": ["entailment", "neutral", "contradiction"],
        },
        "rte": {
            "glue_name": "rte",
            "text_fields": ["sentence1", "sentence2"],
            "num_labels": 2,
            "label_names": ["entailment", "not_entailment"],
        },
    }
    
    def __init__(
        self,
        task: str = "sst2",
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        if task not in self.TASK_CONFIG:
            raise ValueError(f"Unknown task: {task}. Choose from {list(self.TASK_CONFIG.keys())}")
        
        self.task = task
        self.config = self.TASK_CONFIG[task]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self):
        """Load and tokenize dataset."""
        print(f"Loading GLUE/{self.task}...")
        dataset = load_dataset("glue", self.config["glue_name"])
        
        text_fields = self.config["text_fields"]
        
        def tokenize_fn(examples):
            if len(text_fields) == 1:
                return self.tokenizer(
                    examples[text_fields[0]],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                )
            else:
                return self.tokenizer(
                    examples[text_fields[0]],
                    examples[text_fields[1]],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                )
        
        remove_cols = [c for c in dataset["train"].column_names if c not in ["label"]]
        
        print("Tokenizing...")
        self.train_dataset = dataset["train"].map(
            tokenize_fn, batched=True, remove_columns=remove_cols
        )
        
        val_key = "validation_matched" if self.task == "mnli" else "validation"
        self.val_dataset = dataset[val_key].map(
            tokenize_fn, batched=True, remove_columns=remove_cols
        )
        
        self.train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        self.val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        print(f"  {self.task.upper()} loaded!")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Classes: {self.config['num_labels']} ({', '.join(self.config['label_names'])})")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def get_glue_dataloaders(task="sst2", batch_size=32, max_length=128):
    """Convenience function to get GLUE dataloaders."""
    dm = GLUEDataModule(task=task, batch_size=batch_size, max_length=max_length)
    dm.setup()
    return dm.train_dataloader(), dm.val_dataloader(), dm.config["num_labels"]


# Test
if __name__ == "__main__":
    for task in ["sst2", "mnli", "rte"]:
        print(f"\n{'='*50}")
        print(f"Testing {task.upper()}...")
        dm = GLUEDataModule(task=task, batch_size=4, max_length=64)
        dm.setup()
        
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        print(f"  Input shape: {batch['input_ids'].shape}")
        print(f"  Labels: {batch['label']}")
    
    print(f"\n{'='*50}")
    print("All GLUE tasks loaded successfully!")
