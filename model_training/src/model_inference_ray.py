# NOTE: do not use loguru in the actor, will fail to init
#       use plain print instead
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import ray
import torch
from loguru import logger
from ray import serve
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from funcs.utils import find_project_root, now

ROOT = find_project_root()
BASE_MODEL_NAME = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
MAX_LENGTH = 32
MODEL_PATH = ROOT / "models" / "efo_mk1" / "model" / "pytorch_model.bin"
CONFIG_PATH = MODEL_PATH.parent / "config.json"
DEFAULT_OUTPUT_PATH = ROOT / "output" / "efo_mk1_inference.csv"
SAMPLE_FRAC = 0.5
SAMPLE_N = 10_000
BATCH_SIZE = 4_000
LOG_STEP = 10


assert MODEL_PATH.exists()
assert CONFIG_PATH.exists()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--output-path", type=Path, default=DEFAULT_OUTPUT_PATH
    )
    parser.add_argument("-j", "--num-workers", type=int, default=4)
    parser.add_argument("--trial", default=False, action="store_true")
    return parser


class BlueBertInference:
    def __init__(self):
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        self.config = AutoConfig.from_pretrained(
            str(CONFIG_PATH), num_labels=1
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH), config=self.config
        )
        self.model.to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        print("Model init complete.")

    def inference(self, text_1: List[str], text_2: List[str]) -> List[float]:
        encodings = self.tokenizer(
            text_1,
            text_2,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            output = self.model(**encodings)["logits"].reshape(-1).tolist()
        return output

    def __call__(
        self, request_data: serve.utils.ServeRequest,
    ) -> pd.DataFrame:
        text_1 = request_data.query_params["text_1"]
        text_2 = request_data.query_params["text_2"]
        scores = self.inference(text_1, text_2)
        res = pd.DataFrame(
            {"text_1": text_1, "text_2": text_2, "score": scores}
        )
        return res


def make_index_chunks(
    index_list: List[int], batch_size: int
) -> List[List[int]]:
    chunks = [
        index_list[_ : (_ + batch_size)]
        for _ in range(0, len(index_list), batch_size)
    ]
    return chunks


def request_batch(
    idx: int, df: pd.DataFrame, index_chunks: List[List[int]], serve_handle
) -> pd.DataFrame:
    if idx % LOG_STEP == 0:
        print(f"{now()} idx: {idx}")
    idx_list = index_chunks[idx]
    df_sample = df.iloc[idx_list, :]
    res = serve_handle.remote(
        text_1=df_sample["text_1"].tolist(),
        text_2=df_sample["text_2"].tolist(),
    )
    return res


def main():
    parser = create_parser()
    args = parser.parse_args()
    print(args)

    # init ray
    client = serve.start()
    backend_config = {"num_replicas": args.num_workers}
    ray_actor_options = {"num_gpus": 1}
    client.create_backend(
        "bluebert",
        BlueBertInference,
        config=backend_config,
        ray_actor_options=ray_actor_options,
    )
    client.create_endpoint(
        "inference", backend="bluebert", route="/inference", methods=["POST"]
    )

    # load df
    logger.info(f"Load data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    len_df = len(df)
    logger.info(f"len df: {len_df}")
    if args.trial:
        logger.info("Test a smaller scale set")
        frac_n = round(len_df * SAMPLE_FRAC)
        sample_n = min(SAMPLE_N, frac_n)
        logger.info(f"sample: {sample_n}")
        df = df.sample(n=sample_n).reset_index(drop=True)

    # real thing
    index_chunks: List[List[int]] = make_index_chunks(
        df.index.tolist(), batch_size=BATCH_SIZE
    )
    logger.info(f"Split into {len(index_chunks)} chunks")
    logger.info("Compute starts")
    serve_handle = client.get_handle("inference")
    result_batch = ray.get(
        [
            request_batch(
                _, df=df, index_chunks=index_chunks, serve_handle=serve_handle
            )
            for _ in range(len(index_chunks))
        ]
    )
    logger.info("Compute completes")

    # merge
    df_out = pd.concat([pd.DataFrame(_) for _ in result_batch]).reset_index(
        drop=True
    )
    logger.info(f"Write to {args.output_path}")
    df_out.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
