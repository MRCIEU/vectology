import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast

from funcs.efo.efo_bert_model import Model
from funcs.utils import find_project_root

ROOT = find_project_root()
DEFAULT_MODEL_PATH = ROOT / "cache" / "efo_bert" / "models" / "default.ckpt"
DEFAULT_OUTPUT_PATH = ROOT / "output" / "efo_bert_inference.csv"
MAX_TOKENIZATION_LENGTH = 64
SAMPLE_FRAC = 0.5
SAMPLE_N = 10_000
BATCH_SIZE = 1_100


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--output-path", type=Path, default=DEFAULT_OUTPUT_PATH
    )
    parser.add_argument("--trial", default=False, action="store_true")
    return parser


class EfoPandasDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        res: int = len(self.df)
        return res

    def __getitem__(self, i: int):
        res: pd.DataFrame = self.df.iloc[[i]]
        return res


@dataclass
class EfoDataCollator:
    tokenizer: BertTokenizerFast
    max_tokenization_length: int = 128

    def __call__(self, df_list: List[pd.DataFrame]) -> Dict[str, torch.Tensor]:
        df = pd.concat(df_list)
        features = self._get_features(df)
        return features

    def _get_features(self, df: pd.DataFrame) -> Dict:
        encodings = self.tokenizer(
            df["text_1"].tolist(),
            df["text_2"].tolist(),
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
        }
        return res


def main():
    parser = create_parser()
    args = parser.parse_args()
    logger.info(args)
    assert args.model_path.exists()
    assert args.data_path.exists()

    logger.info(f"Load model from {args.model_path}")
    model = Model().load_from_checkpoint(str(args.model_path)).cuda().eval()
    tokenizer = model.bert_tokenizer

    logger.info(f"Load data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    len_df = len(df)
    logger.info(f"len df: {len_df}")
    if args.trial:
        logger.info("Test a smaller scale set")
        frac_n = round(len_df * SAMPLE_FRAC)
        sample_n = min(SAMPLE_N, frac_n)
        logger.info(f"sample: {sample_n}")
        df = df.sample(n=sample_n)
    dataset = EfoPandasDataset(df)
    collate_fn = EfoDataCollator(
        tokenizer=tokenizer, max_tokenization_length=MAX_TOKENIZATION_LENGTH
    )
    data_loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
    )

    logger.info("Compute begins")
    scores = []
    for data_batch in tqdm(iter(data_loader)):
        with torch.no_grad():
            outputs = model(**data_batch)
            logits = outputs["logits"].reshape(-1).tolist()
            scores = scores + logits
    logger.info(f"len scores: {len(scores)}")

    output_df = df.assign(score=scores)
    output_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
