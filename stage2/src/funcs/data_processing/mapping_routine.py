from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from icecream import ic
from loguru import logger

from funcs import info


def prep_trait_efo_mapping_agg(
    ebi_data: pd.DataFrame,
    model_collection: Dict[str, info.ModelInfo],
    batet_score: float,
    verbose: bool = False,
    tsv_p: bool = False,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    - tsv_p: True if the cache is in tsv else csv
    - cache_dir: dir to store intermediate cache
    """
    res_items = {}
    top_100_cache = {k: v["top_100"] for k, v in model_collection.items()}
    mapping_types = ["Exact", "Broad", "Narrow"]
    manual_df = ebi_data
    manual_df.loc[
        ~manual_df["MAPPING_TYPE"].isin(mapping_types), "MAPPING_TYPE"
    ] = "Other"
    item = manual_df["MAPPING_TYPE"].value_counts().pipe(dict)
    res_items["Manual"] = item

    for k, v in top_100_cache.items():
        df = trait_efo_mapping_results(
            cache_path=v,
            model_name=k,
            manual_df=manual_df,
            mapping_types=mapping_types,
            batet_score=batet_score,
            verbose=verbose,
            tsv_p=tsv_p,
        )
        if cache_dir is not None:
            cache_path = cache_dir / f"{k}_{batet_score}.csv"
            logger.info(f"Write to {cache_path}")
            df.to_csv(cache_path, index=False)
        item = df["MAPPING_TYPE"].value_counts().pipe(dict)
        res_items[k] = item

    res = (
        pd.DataFrame.from_dict(res_items, orient="index")
        .fillna(0)
        .assign(Total=lambda df: df["Exact"] + df["Broad"] + df["Narrow"] + df["Other"])
        .rename_axis("Model")
        .sort_values(by="Total", ascending=False)
        .reset_index(drop=False)
    )
    return res


def trait_efo_mapping_results(
    cache_path: Path,
    model_name: str,
    manual_df: pd.DataFrame,
    mapping_types: List[str],
    batet_score: float,
    verbose: bool = False,
    tsv_p: bool = False,
) -> pd.DataFrame:
    assert cache_path.exists(), cache_path
    df = pd.read_csv(cache_path) if not tsv_p else pd.read_csv(cache_path, sep="\t")
    if verbose:
        ic(df.shape)
    df = df.merge(
        manual_df[["mapping_id", "MAPPING_TYPE"]], on="mapping_id"
    ).drop_duplicates(subset=["mapping_id", "manual"])
    if verbose:
        ic(df.shape)
    df.loc[~df["MAPPING_TYPE"].isin(mapping_types), "MAPPING_TYPE"] = "Other"
    df = df[df["nx"] >= batet_score]
    if verbose:
        ic(df.shape)
    return df


def prep_weighted_average_df(model_collection, top_num: int, ebi_df) -> pd.DataFrame:
    # zooma has no res above 1
    if top_num > 1:
        allowed_model_coll = {k: v for k, v in model_collection.items() if k != "Zooma"}
    else:
        allowed_model_coll = model_collection
    weighted_average_results = {
        k: calc_weighted_average(
            model_name=k,
            cache_path=v["top_100"],
            ebi_df=ebi_df,
            top_num=top_num,
        )
        for k, v in allowed_model_coll.items()
    }
    df = pd.DataFrame(weighted_average_results).assign(efo=ebi_df["full_id"])
    df_melt = pd.melt(df, id_vars=["efo"]).rename(columns={"variable": "Model"})
    return df_melt


def calc_weighted_average(model_name, cache_path, ebi_df, top_num: int):
    logger.info(model_name)
    mapping_types_all = ["Exact", "Broad", "Narrow"]
    # df = pd.read_csv(cache_path, sep="\t")
    df = pd.read_csv(cache_path)
    ebi_df = ebi_df[ebi_df["MAPPING_TYPE"].isin(mapping_types_all)]

    manual_efos = ebi_df["full_id"].tolist()
    res = []
    for _ in manual_efos:
        nx_scores = df[df["manual"] == _].head(n=top_num)["nx"].tolist()
        weights = list(reversed(range(1, (len(nx_scores) + 1))))
        try:
            weighted_avg = round(np.average(nx_scores, weights=weights), 3)
        except:
            weighted_avg = 0
        res.append(weighted_avg)
    logger.info(len(res))
    return res


def prep_weighted_average_df_new(model_collection, top_num: int) -> pd.DataFrame:
    def _calc_wa(df, top_num):
        nx_scores = df["nx"].head(top_num).tolist()
        weights = list(reversed(range(1, (len(nx_scores) + 1))))
        try:
            weighted_avg = round(np.average(nx_scores, weights=weights), 3)
        except:
            weighted_avg = 0
        res_df = pd.Series({"value": weighted_avg})
        return res_df

    def _calc_wa_df(model_name, cache_path, top_num: int):
        logger.info(model_name)
        assert cache_path.exists(), cache_path
        df = pd.read_csv(cache_path)
        df = (
            df.groupby("mapping_id")
            .apply(lambda df: _calc_wa(df, top_num))
            .reset_index(drop=False)
            .assign(Model=model_name)
        )
        return df

    # zooma has no res above 1
    if top_num > 1:
        allowed_model_coll = {k: v for k, v in model_collection.items() if k != "Zooma"}
    else:
        allowed_model_coll = model_collection
    res = pd.concat(
        [
            _calc_wa_df(
                model_name=k,
                cache_path=v["top_100"],
                top_num=top_num,
            )
            for k, v in allowed_model_coll.items()
        ]
    ).reset_index(drop=True)
    return res


def get_trait_efo_unmapped(
    ebi_data: pd.DataFrame,
    model_collection: Dict[str, info.ModelInfo],
    verbose: bool = False,
    tsv_p: bool = False,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    def _get_unmapped(
        cache_path: Path,
        model_name: str,
        manual_df: pd.DataFrame,
        mapping_types: List[str],
        verbose: bool = False,
        tsv_p: bool = False,
    ) -> pd.DataFrame:
        df = pd.read_csv(cache_path) if not tsv_p else pd.read_csv(cache_path, sep="\t")
        if verbose:
            ic(df.shape)
        df = df.assign(
            group_rank=lambda df: df.groupby(["mapping_id"]).cumcount(ascending=True)
        )
        df = df[df["nx"] >= 1.0]
        # df = df.merge(
        #     manual_df[["mapping_id", "MAPPING_TYPE"]], on="mapping_id"
        # ).drop_duplicates(subset=["mapping_id", "manual"])
        df = manual_df.merge(df[["mapping_id", "nx", "group_rank"]], on="mapping_id")
        if verbose:
            ic(df.shape)
        df.loc[~df["MAPPING_TYPE"].isin(mapping_types), "MAPPING_TYPE"] = "Other"
        df = df.assign(model_name=model_name)
        if verbose:
            ic(df.shape)
        return df

    assert output_dir is not None and output_dir.exists(), output_dir

    top_100_cache = {k: v["top_100"] for k, v in model_collection.items()}
    mapping_types = ["Exact", "Broad", "Narrow"]
    manual_df = ebi_data
    manual_df.loc[
        ~manual_df["MAPPING_TYPE"].isin(mapping_types), "MAPPING_TYPE"
    ] = "Other"

    df_list = []
    for k, v in top_100_cache.items():
        df = _get_unmapped(
            cache_path=v,
            model_name=k,
            manual_df=manual_df,
            mapping_types=mapping_types,
            verbose=verbose,
            tsv_p=tsv_p,
        )
        df_list.append(df)

        output_path = output_dir / f"{k}.csv"
        logger.info(f"Write to {output_path}")
        df.to_csv(output_path, index=False)
    df_agg = pd.concat(df_list)

    return df_agg
