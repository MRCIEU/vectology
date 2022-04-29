from pathlib import Path

import networkx as nx
import pandas as pd
from loguru import logger
from nxontology import NXOntology

from funcs import paths, utils
from funcs.data_processing import stage1_nx


def get_efo_nodes(verbose: bool = True) -> pd.DataFrame:
    efo_nodes_path = paths.init["efo_nodes"]
    # get EFO node data
    df = pd.read_csv(efo_nodes_path)
    df.rename(columns={"efo.value": "efo_label", "efo.id": "efo_id"}, inplace=True)
    # drop type
    df.drop(["efo.type"], inplace=True, axis=1)
    # lowercase the label
    df["efo_label"] = df["efo_label"].str.lower()
    # drop duplicates by name
    df.drop_duplicates(subset=["efo_label"], inplace=True)
    if verbose:
        logger.info(utils.df_info(df))
    return df


def get_ebi_data(verbose: bool = True) -> pd.DataFrame:
    file_path = paths.stage1["ebi_ukb_cleaned"]
    df = pd.read_csv(file_path)
    if verbose:
        logger.info(utils.df_info(df))
        logger.info(f'\n{df["MAPPING_TYPE"].value_counts()}')
    return df


def get_efo_nx(verbose: bool = True) -> NXOntology:
    file_path = paths.stage2["efo_nx"]
    if file_path.exists():
        if verbose:
            logger.info(f"read from cache {file_path}")
        efo_nx = nx.read_gpickle(file_path)
    else:
        if verbose:
            logger.info(f"write to {file_path}")
        efo_rels = pd.read_csv(paths.init["efo_edges"])
        efo_nx = stage1_nx.create_nx(efo_rel_df=efo_rels)
        nx.write_gpickle(efo_nx, file_path)
    return efo_nx


def get_top100_using_pairwise_file(
    cache_path: Path, efo_nx: NXOntology, ebi_df: pd.DataFrame
) -> pd.DataFrame:
    df = pd.read_csv(cache_path, sep="\t")
    top_num = 100
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
