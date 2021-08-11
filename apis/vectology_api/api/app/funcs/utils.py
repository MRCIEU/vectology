from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


def cosine_sim(x, y) -> Optional[float]:
    res = 1 - cosine(np.array(x), np.array(y))
    if np.isnan(res):
        return None
    return res


def pairwise_cosine_sim(pair, embeddings):
    idx1 = pair[0]
    idx2 = pair[1]
    res = {
        "idx1": idx1,
        "idx2": idx2,
        "cosine_similarity": cosine_sim(embeddings[idx1], embeddings[idx2]),
    }
    return res


def complete_pairs(pairs):
    reversed_pairs = [
        {
            "idx1": item["idx2"],
            "idx2": item["idx1"],
            "cosine_similarity": item["cosine_similarity"],
        }
        for item in pairs
    ]
    id_set = set(
        [item["idx1"] for item in pairs] + [item["idx2"] for item in pairs]
    )
    self_pairs = [
        {"idx1": item, "idx2": item, "cosine_similarity": 1.0}
        for item in list(id_set)
    ]
    full_pairs = pairs + reversed_pairs + self_pairs
    return full_pairs


def nest_pairs(pairs):
    df = (
        pd.DataFrame(complete_pairs(pairs))
        .sort_values(by=["idx1", "idx2"])
        .assign(
            idx1=lambda df: df["idx1"].astype(str),
            idx2=lambda df: df["idx2"].astype(str),
        )
        .replace({np.nan: None})
    )
    df_nested = (
        df.groupby(["idx1"])
        .apply(
            lambda df: (
                df.drop(columns=["idx1"])
                .rename(columns={"idx2": "x", "cosine_similarity": "y"})
                .to_dict(orient="records")
            )
        )
        .to_frame(name="data")
    )
    df_nested.index.names = ["name"]
    nested = df_nested.reset_index().to_dict(orient="records")
    return nested
