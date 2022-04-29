from string import punctuation
from typing import List

import numpy as np
import pandas as pd
import sent2vec
from loguru import logger
from nltk import word_tokenize
from nltk.corpus import stopwords
from nxontology import NXOntology
from scipy.spatial import distance

from funcs import paths
from funcs.data_processing import stage1_processing
from funcs.utils import ic

ECHO_STEP = 200


def preprocess_sentence(text: str) -> str:

    stop_words = set(stopwords.words("english"))
    text = text.replace("/", " / ")
    text = text.replace(".-", " .- ")
    text = text.replace(".", " . ")
    text = text.replace("'", " ' ")
    text = text.lower()

    tokens = [
        token
        for token in word_tokenize(text)
        if token not in punctuation and token not in stop_words
    ]

    return " ".join(tokens)


def biosentvec_encode_terms(
    text_list: List[str], biosentvec_model: sent2vec.Sent2vecModel
) -> List[np.ndarray]:
    def _embed(idx: int, total: int, text: str) -> np.ndarray:
        if idx % ECHO_STEP == 0:
            logger.info(f"#{idx} / {total}")
        preprocessed = preprocess_sentence(text)
        embed = biosentvec_model.embed_sentence(preprocessed)
        res = embed[0]
        return res

    res = [
        _embed(idx=idx, total=len(text_list), text=_) for idx, _ in enumerate(text_list)
    ]
    return res


def calc_pairs(ebi_encode: np.ndarray, efo_encode: np.ndarray) -> np.ndarray:
    res = distance.cdist(ebi_encode, efo_encode, "cosine")
    return res


def make_pairwise_flat(
    pairwise_data: np.ndarray, ebi_df: pd.DataFrame, efo_node_df: pd.DataFrame
) -> pd.DataFrame:
    ebi_efo_list = ebi_df["full_id"].tolist()
    efo_list = efo_node_df["efo_id"].tolist()
    res_list = []
    for idx_i, ebi_efo_item in enumerate(ebi_efo_list):
        if idx_i % ECHO_STEP == 0:
            logger.info(f"#{idx_i} / {len(ebi_efo_list)}")
        for idx_j, efo_item in enumerate(efo_list):
            score = 1 - pairwise_data[idx_i][idx_j]
            item = {
                "mapping_id": idx_i + 1,
                "manual": ebi_efo_item,
                "prediction": efo_item,
                "score": score,
            }
            res_list.append(item)
    res_df = pd.DataFrame(res_list)
    return res_df


def make_top100(
    pairwise_flat: pd.DataFrame, efo_nx: NXOntology, ebi_df: pd.DataFrame
) -> pd.DataFrame:
    # first do `filter_paiwise_file`
    top_num = 100
    df = (
        pairwise_flat.sort_values(by=["score"], ascending=False)
        .groupby("mapping_id")
        .head(top_num)
        .drop_duplicates(subset=["mapping_id", "manual", "prediction"])
        .sort_values(by=["mapping_id", "score"], ascending=[True, False])
    )
    # then do `get_top_using_pairwise_file`
    top_res = []
    for idx_i, row_i in ebi_df.iterrows():
        mapping_id = row_i["mapping_id"]
        efo_predictions = df[df["mapping_id"] == mapping_id].head(n=top_num)[
            ["prediction", "score"]
        ]
        for idx_j, row_j in efo_predictions.iterrows():
            manual_efo = row_i["full_id"]
            predicted_efo = row_j["prediction"]
            score = row_j["score"]
            try:
                res = efo_nx.similarity(manual_efo, predicted_efo).results()
                nx_val = res["batet"]
            except:
                nx_val = 0
            top_res.append(
                {
                    "mapping_id": idx_i + 1,
                    "manual": row_i["full_id"],
                    "prediction": predicted_efo,
                    "score": score,
                    "nx": nx_val,
                }
            )
    res_df = pd.DataFrame(top_res)
    return res_df


def main():

    biosentvec_model_path = paths.init["biosentvec_model"]
    assert biosentvec_model_path.exists()

    ebi_df = stage1_processing.get_ebi_data()
    efo_node_df = stage1_processing.get_efo_nodes()
    efo_nx = stage1_processing.get_efo_nx()

    ebi_encode_path = paths.stage2["output"] / "biosentvec-ebi-encode.npy"
    efo_node_encode_path = paths.stage2["output"] / "biosentvec-efo-encode.npy"
    pairwise_results_path = paths.stage2["output"] / "biosentvec-dd.npy"
    pairwise_flat_path = pairwise_results_path.parent / "biosentvec-pairwise.csv.gz"
    top100_path = paths.stage2["output"] / "biosentvec-top-100.csv"

    if not ebi_encode_path.exists() or not efo_node_encode_path.exists():
        logger.info("Init biosentvec model")
        biosentvec_model = sent2vec.Sent2vecModel()
        biosentvec_model.load_model(str(biosentvec_model_path))

        logger.info("Encode ebi data")
        if not ebi_encode_path.exists():
            ebi_encode = biosentvec_encode_terms(
                text_list=ebi_df["query"].tolist(), biosentvec_model=biosentvec_model
            )
            np.save(str(ebi_encode_path), ebi_encode)

        logger.info("Encode efo data")
        if not efo_node_encode_path.exists():
            efo_encode = biosentvec_encode_terms(
                text_list=efo_node_df["efo_label"].tolist(),
                biosentvec_model=biosentvec_model,
            )
            np.save(str(efo_node_encode_path), efo_encode)
    else:
        ebi_encode = np.load(str(ebi_encode_path))
        efo_encode = np.load(str(efo_node_encode_path))
    ic(ebi_encode.shape)
    ic(efo_encode.shape)

    if not pairwise_results_path.exists():
        pairwise_results = calc_pairs(ebi_encode=ebi_encode, efo_encode=efo_encode)
        np.save(str(pairwise_results_path), pairwise_results)
    else:
        pairwise_results = np.load(pairwise_results_path)
    ic(pairwise_results.shape)

    if not pairwise_flat_path.exists():
        pairwise_flat = make_pairwise_flat(
            pairwise_data=pairwise_results, ebi_df=ebi_df, efo_node_df=efo_node_df
        )
        pairwise_flat.to_csv(pairwise_flat_path, index=False)
    else:
        pairwise_flat = pd.read_csv(pairwise_flat_path)
    ic(pairwise_flat.shape)

    if not top100_path.exists():
        top100_results = make_top100(
            pairwise_flat=pairwise_flat, efo_nx=efo_nx, ebi_df=ebi_df
        )
        top100_results.to_csv(top100_path, index=False)
    else:
        top100_results = pd.read_csv(top100_path)
    ic(top100_results.shape)


if __name__ == "__main__":
    main()
