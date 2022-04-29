import json
from typing import List

import numpy as np
import pandas as pd
import sent2vec
from gensim.models import KeyedVectors
from loguru import logger
from metaflow import FlowSpec, Parameter, step
from pydash import py_
from scipy.spatial import distance

from funcs import paths, utils
from funcs.data_processing import mapping_routine, pairwise_routine, stage1_processing
from funcs.nlp import nlp

from icecream import ic  # noqa


proj_root = utils.find_project_root()
data_root = utils.find_data_root()

ECHO_STEP = 200
MODEL_NAME = "biosentvec-bioconceptvec-skipgram"
OUTPUT_PATH = paths.stage2["output"]
PUBTATOR_DIR = data_root / "output" / "pubtator"
assert PUBTATOR_DIR.exists()

BIOSENTVEC_MODEL_PATH = paths.init["biosentvec_model"]
assert BIOSENTVEC_MODEL_PATH.exists()

BIOCONCEPTVEC_MODEL_PATH = (
    proj_root / "models" / "bioconceptvec" / "bioconceptvec_word2vec_skipgram.bin"
)
assert BIOCONCEPTVEC_MODEL_PATH.exists()


def bioconceptvec_augment(
    base_vectors: np.ndarray,
    ner_df: pd.DataFrame,
    bioconceptvec_embeddings: KeyedVectors,
    vocab_list: List[str],
) -> np.ndarray:
    total = base_vectors.shape[0]
    for idx in range(total):
        if idx % ECHO_STEP == 0:
            logger.info(f"#{idx} / {total}")
        mapping_id = idx + 1
        ner_res = py_.flatten(
            ner_df[ner_df["mapping_id"] == mapping_id]["ner_res"].tolist()
        )
        if len(ner_res) == 0:
            continue
        embedding_res = [
            bioconceptvec_embeddings[_]
            for _ in ner_res
            if isinstance(_, str) and _ in vocab_list
        ]
        base_vectors[idx,] = nlp.harmonize_vectors(  # noqa: E231
            main_vector=base_vectors[
                idx,
            ],
            addons=embedding_res,
        )
    return base_vectors


class BioconceptvecFlow(FlowSpec):

    OVERWRITE = Parameter(
        "overwrite",
        help="overwrite",
        default=False,
        is_flag=True,
    )

    @step
    def start(self):

        self.ebi_df = stage1_processing.get_ebi_data()
        self.efo_node_df = stage1_processing.get_efo_nodes()
        self.efo_nx = stage1_processing.get_efo_nx()
        ner_res_path = PUBTATOR_DIR / "ner_res.json"
        assert ner_res_path.exists()
        with ner_res_path.open() as f:
            self.ner_df = pd.DataFrame(json.load(f))
        self.next(self.make_encode)

    @step
    def make_encode(self):

        ebi_encode_path = OUTPUT_PATH / f"{MODEL_NAME}-ebi-encode.npy"
        efo_node_encode_path = OUTPUT_PATH / f"{MODEL_NAME}-efo-encode.npy"
        if (
            not ebi_encode_path.exists()
            or not efo_node_encode_path.exists()
            or self.OVERWRITE
        ):
            self.ebi_encode, self.efo_encode = self._make_encode()
            np.save(str(ebi_encode_path), self.ebi_encode)
            np.save(str(efo_node_encode_path), self.efo_encode)
        else:
            print("Read from cache")
            self.ebi_encode = np.load(str(ebi_encode_path), allow_pickle=True)
            self.efo_encode = np.load(str(efo_node_encode_path), allow_pickle=True)
        self.next(self.make_pairwise)

    @step
    def make_pairwise(self):
        pairwise_results_path = OUTPUT_PATH / f"{MODEL_NAME}-dd.npy"
        if not pairwise_results_path.exists() or self.OVERWRITE:
            self.pairwise_results = distance.cdist(
                self.ebi_encode, self.efo_encode, "cosine"
            )
            np.save(str(pairwise_results_path), self.pairwise_results)
        else:
            self.pairwise_results = np.load(pairwise_results_path)
        self.next(self.make_pairwise_flat)

    @step
    def make_pairwise_flat(self):
        pairwise_flat_path = OUTPUT_PATH / f"{MODEL_NAME}-pairwise.csv.gz"
        if not pairwise_flat_path.exists() or self.OVERWRITE:
            self.pairwise_flat = pairwise_routine.make_pairwise_flat(
                pairwise_data=self.pairwise_results,
                ebi_df=self.ebi_df,
                efo_node_df=self.efo_node_df,
            )
            self.pairwise_flat.to_csv(pairwise_flat_path, index=False)
        else:
            self.pairwise_flat = pd.read_csv(pairwise_flat_path)
        self.next(self.make_top100)

    @step
    def make_top100(self):
        top100_path = OUTPUT_PATH / f"{MODEL_NAME}-top-100.csv"
        if not top100_path.exists() or self.OVERWRITE:
            self.top100_results = pairwise_routine.make_top100(
                pairwise_flat=self.pairwise_flat, efo_nx=self.efo_nx, ebi_df=self.ebi_df
            )
            self.top100_results.to_csv(top100_path, index=False)
        else:
            self.top100_results = pd.read_csv(top100_path)
        self.next(self.verify)

    @step
    def verify(self):

        model_collection = {
            "BioSentVec-BioConceptVec-skipgram": {
                "model": "BioSentVec-BioConceptVec-skipgram",
                "stage": "stage2",
                "top_100": OUTPUT_PATH / f"{MODEL_NAME}-top-100.tsv.gz",
                "pairwise_filter": None,
                "top_100_revised": OUTPUT_PATH / f"{MODEL_NAME}-top-100.csv",
            },
        }

        trait_efo_mapping_df = mapping_routine.prep_trait_efo_mapping_agg(
            ebi_data=self.ebi_df, model_collection=model_collection, batet_score=1.0
        )
        print(trait_efo_mapping_df)

        self.next(self.end)

    @step
    def end(self):
        pass

    def _make_encode(self):
        print("Init biosentvec model")
        biosentvec_model = sent2vec.Sent2vecModel()
        biosentvec_model.load_model(str(BIOSENTVEC_MODEL_PATH))

        print("Init bioconceptvec model")
        bioconceptvec_embeddings = KeyedVectors.load_word2vec_format(
            str(BIOCONCEPTVEC_MODEL_PATH), binary=True
        )

        print("Encode ebi")
        ebi_encode_base = nlp.biosentvec_encode_terms(
            text_list=self.ebi_df["query"].tolist(),
            biosentvec_model=biosentvec_model,
        )

        print("Encode efo")
        efo_encode_base = nlp.biosentvec_encode_terms(
            text_list=self.efo_node_df["efo_label"].tolist(),
            biosentvec_model=biosentvec_model,
        )

        print("Add bioconceptvec")
        ebi_ner_df = self.ner_df[self.ner_df["batch"] == "ebi"].reset_index(drop=True)
        efo_ner_df = self.ner_df[self.ner_df["batch"] == "efo"].reset_index(drop=True)
        vocab_list = list(bioconceptvec_embeddings.key_to_index.keys())
        ebi_encode = bioconceptvec_augment(
            base_vectors=ebi_encode_base,
            ner_df=ebi_ner_df,
            bioconceptvec_embeddings=bioconceptvec_embeddings,
            vocab_list=vocab_list,
        )
        efo_encode = bioconceptvec_augment(
            base_vectors=efo_encode_base,
            ner_df=efo_ner_df,
            bioconceptvec_embeddings=bioconceptvec_embeddings,
            vocab_list=vocab_list,
        )

        return ebi_encode, efo_encode


if __name__ == "__main__":
    BioconceptvecFlow()
