import argparse
import math
import sqlite3
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

from funcs.hyperparams import DataModuleHparams, add_data_module_specific_args
from funcs.utils import find_project_root

ROOT = find_project_root()
DATA_DIR = ROOT / "output" / "efo-gwas"
SQL_CHUNKSIZE = 20
# due to sql loader which loads a large chunk of entries,
# batch size should be smaller
DATA_BATCH_SIZE = 12
MAX_TOKENIZATION_LENGTH = 32


@dataclass
class EfoDataModuleHparams(DataModuleHparams):
    sql_chunksize: int = SQL_CHUNKSIZE

    def __init__(self, **kwargs):
        super().__init__()
        names = set([f.name for f in fields(self)])
        for key, value in kwargs.items():
            if key in names:
                setattr(self, key, value)


class DataModule(pl.LightningDataModule):
    def __init__(
        self, args: Optional[argparse.Namespace] = None,
    ):
        super().__init__()
        if args is not None:
            self.hparams = asdict(EfoDataModuleHparams(**vars(args)))  # type: ignore
        else:
            self.hparams = asdict(EfoDataModuleHparams())  # type: ignore
        logger.info(f"data module hparams: {self.hparams}")
        model_name: str = self.hparams["model_name"]  # type: ignore
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.collate_fn = EfoDataCollator(
            tokenizer=self.tokenizer,
            max_tokenization_length=self.hparams["max_tokenization_length"],
        )

    def setup(self, stage: Optional[str] = "fit"):
        if stage == "fit" or stage is None:
            train_db_path = DATA_DIR / "distance_train.db"
            val_db_path = DATA_DIR / "distance_val.db"
            meta_file_path = DATA_DIR / "distance_meta.csv"
            meta_df = pd.read_csv(meta_file_path)
            train_size = meta_df[meta_df["table"] == "TRAIN"][
                "count"
            ].to_list()[0]
            val_size = meta_df[meta_df["table"] == "VALIDATION"][
                "count"
            ].to_list()[0]
            if self.hparams["train_sample"] != 1.0:
                train_size = math.floor(
                    train_size * self.hparams["train_sample"]
                )
            if self.hparams["val_sample"] != 1.0:
                val_size = math.floor(val_size * self.hparams["val_sample"])
            self.train_dataset = EfoSqlDataset(
                db_path=train_db_path,
                name="TRAIN",
                total_size=train_size,
                chunksize=self.hparams["sql_chunksize"],
            )
            self.val_dataset = EfoSqlDataset(
                db_path=val_db_path,
                name="VALIDATION",
                total_size=val_size,
                chunksize=self.hparams["sql_chunksize"],
            )

    def train_dataloader(self):
        data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["train_batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=True,
        )
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams["eval_batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        return data_loader

    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = add_data_module_specific_args(parent_parser)
        parser.add_argument("--sql_chunksize", default=SQL_CHUNKSIZE, type=int)
        parser.add_argument(
            "--train_batch_size", default=DATA_BATCH_SIZE, type=int
        )
        parser.add_argument(
            "--eval_batch_size", default=DATA_BATCH_SIZE, type=int
        )
        parser.add_argument(
            "--max_tokenization_length",
            default=MAX_TOKENIZATION_LENGTH,
            type=int,
        )
        return parser


class EfoSqlDataset(Dataset):
    def __init__(
        self, db_path: Path, name: str, total_size: int, chunksize: int = 5
    ):

        self.total_size = total_size
        self.chunk_size = chunksize
        self.name = name
        self._len = self._get_len()
        self.db_path = db_path

    def __len__(self):
        return self._len

    def _get_len(self):
        return math.floor(self.total_size / self.chunk_size)

    def _get_record_range(self, idx: int) -> Tuple:
        start = idx * self.chunk_size
        end = (idx + 1) * self.chunk_size
        return tuple(range(start, end))

    def __getitem__(self, i: int) -> pd.DataFrame:
        record_range = self._get_record_range(idx=i)
        query = """
        SELECT idx, source, target, distance from {table_name}
        WHERE idx in {record_range}
        ORDER BY idx;
        """.replace(
            "\n", " "
        ).format(
            table_name=self.name, record_range=str(record_range)
        )
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn)
            return df


@dataclass
class EfoDataCollator:
    tokenizer: BertTokenizerFast
    max_tokenization_length: int = 128

    def __call__(self, df_list: List[pd.DataFrame]) -> Dict[str, torch.Tensor]:
        df = pd.concat(df_list)
        # sanitize df
        df = df.dropna()
        features = self._get_features(df)
        return features

    def _get_features(self, df: pd.DataFrame) -> Dict:
        encodings = self.tokenizer(
            df["source"].tolist(),
            df["target"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=self.max_tokenization_length,
        )
        res = {
            "input_ids": torch.tensor(
                encodings["input_ids"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                encodings["attention_mask"], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                encodings["token_type_ids"], dtype=torch.long
            ),
            "label": torch.tensor(df["distance"].tolist(), dtype=torch.float),
        }
        return res
