from typing import Callable, Dict, List, Optional

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast


def flatten_doc_by_sentence(docs: List[str], segmenter) -> List[str]:
    """Split a doc by sentence, then flatten all nested sentences to a list."""
    segmented_docs = [segmenter.segment(item) for item in docs]
    flattened_docs = [sent for doc in segmented_docs for sent in doc]
    return flattened_docs


class TensorDataset(Dataset):
    """A struct to conform with huggingface transformer's data collator"""

    def __init__(self, input_ids: List[Tensor]):

        self.examples = input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Tensor:
        return self.examples[i]


def make_dataloader(
    dataset: Dataset,
    collate_fn: Callable,
    stage: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 1,
) -> DataLoader:
    data_loader_options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
    }
    if stage == "train":
        data_loader_options["shuffle"] = True
    data_loader = DataLoader(**data_loader_options)  # type: ignore
    return data_loader


def make_features(
    docs: List[str],
    tokenizer: BertTokenizerFast,
    max_tokenization_length: int = 128,
) -> Dict:
    """Tokenize a dataset (batch)."""
    encodings = tokenizer(
        docs,
        truncation=True,
        padding="max_length",
        max_length=max_tokenization_length,
    )
    res = {
        "docs": docs,
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "token_type_ids": encodings["token_type_ids"],
    }
    return res


def get_batch_inputs(batch) -> Dict:
    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "token_type_ids": batch["token_type_ids"],
        "labels": batch["label"],
    }
    return inputs
