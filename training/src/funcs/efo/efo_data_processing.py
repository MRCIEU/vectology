import math
import random
import sqlite3
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd
from loguru import logger
from tqdm.contrib.concurrent import process_map


def make_efo_graph(
    efo_node: pd.DataFrame,
    efo_rel: pd.DataFrame,
    cache_file: Path,
    overwrite: bool = False,
) -> nx.DiGraph:
    def _make():
        node_df = (
            efo_node[["efo.id", "efo.value"]]
            .drop_duplicates()
            .rename(columns={"efo.value": "label", "efo.id": "id"})
            .drop_duplicates(subset="label")
        )
        node_dict = node_df.set_index("label").to_dict("index")
        rel_df = efo_rel.merge(
            node_df.rename(columns={"id": "efo.id"}),
            left_on="efo.id",
            right_on="efo.id",
            how="inner",
        ).merge(
            node_df.rename(
                columns={"id": "parent_efo.id", "label": "parent_label"}
            ),
            left_on="parent_efo.id",
            right_on="parent_efo.id",
            how="inner",
        )
        graph = nx.from_pandas_edgelist(
            rel_df,
            source="label",
            target="parent_label",
            create_using=nx.DiGraph,
        )
        nx.set_node_attributes(graph, node_dict)
        return graph

    if not cache_file.exists() or overwrite:
        graph = _make()
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"write to file: {cache_file}")
        nx.write_gpickle(graph, cache_file)
    else:
        logger.info(f"read from file: {cache_file}")
        graph = nx.read_gpickle(cache_file)
    return graph


def make_efo_gwas_graph(
    efo_node: pd.DataFrame,
    efo_rel: pd.DataFrame,
    efo_desc: pd.DataFrame,
    gwas_df: pd.DataFrame,
    cache_file: Path,
    overwrite: bool = False,
) -> Tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    def _make_desc(row):
        synonyms = None
        if row["synonyms"] is not None:
            synonyms = "; ".join(row["synonyms"])
        description = None
        if row["description"] is not None:
            description = row["description"]
        res = "; ".join(
            [_ for _ in [row["label"], synonyms, description] if _ is not None]
        )
        return res

    def _make():
        gwas_df_simple = (
            gwas_df[
                ["DISEASE/TRAIT", "STUDY", "MAPPED_TRAIT", "MAPPED_TRAIT_URI"]
            ]
            .rename(
                columns={
                    "DISEASE/TRAIT": "label",
                    "STUDY": "description",
                    "MAPPED_TRAIT": "efo_label",
                    "MAPPED_TRAIT_URI": "efo_id",
                }
            )
            .drop_duplicates(subset=["label"])
            .dropna()
            .assign(
                desc=lambda df: "; ".join(
                    [
                        _
                        for _ in [df["label"], df["description"]]
                        if _ is not None
                    ]
                )
            )
        )
        efo_desc_simple = efo_desc.reset_index()[
            ["id", "label", "description", "synonyms"]
        ].assign(desc=lambda df: df.apply(_make_desc, axis=1))
        efo_node_df = (
            efo_node[["efo.id", "efo.value"]]
            .drop_duplicates()
            .rename(columns={"efo.value": "label", "efo.id": "id"})
            .merge(
                efo_desc_simple[["id", "desc"]],
                left_on="id",
                right_on="id",
                how="inner",
            )
            .drop_duplicates(subset="label")
        )
        node_df = pd.concat(
            [
                efo_node_df[["label", "desc"]].assign(source="efo"),
                gwas_df_simple[["label", "desc"]].assign(
                    source="gwas-catalog"
                ),
            ]
        ).drop_duplicates(subset=["label"])
        efo_rel_df = efo_rel.merge(
            efo_node_df.rename(columns={"id": "efo.id"}),
            left_on="efo.id",
            right_on="efo.id",
            how="inner",
        ).merge(
            efo_node_df.rename(
                columns={"id": "parent_efo.id", "label": "parent_label"}
            ),
            left_on="parent_efo.id",
            right_on="parent_efo.id",
            how="inner",
        )[
            ["label", "parent_label"]
        ]
        efo_rel_df = efo_rel_df[
            efo_rel_df["label"].isin(
                node_df[node_df["source"] == "efo"]["label"].tolist()
            )
            & efo_rel_df["parent_label"].isin(
                node_df[node_df["source"] == "efo"]["label"].tolist()
            )
        ]
        gwas_rel_df = gwas_df_simple[["label", "efo_label"]].rename(
            columns={"efo_label": "parent_label"}
        )
        gwas_rel_df = gwas_rel_df[
            gwas_rel_df["label"].isin(
                node_df[node_df["source"] == "gwas-catalog"]["label"]
            )
            & gwas_rel_df["parent_label"].isin(
                node_df[node_df["source"] == "efo"]["label"]
            )
        ]
        rel_df = pd.concat([efo_rel_df, gwas_rel_df])
        graph = nx.from_pandas_edgelist(
            rel_df,
            source="label",
            target="parent_label",
            create_using=nx.DiGraph,
        )
        return graph, node_df, rel_df

    node_df_path = cache_file.parent / "node_df.csv"
    rel_df_path = cache_file.parent / "rel_df.csv"
    if (
        not sum(
            [path.exists() for path in [cache_file, node_df_path, rel_df_path]]
        )
        == 3
        or overwrite
    ):
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        graph, node_df, rel_df = _make()
        logger.info(f"write to file: {cache_file}")
        nx.write_gpickle(graph, cache_file)
        node_df.to_csv(node_df_path, index=False)
        rel_df.to_csv(rel_df_path, index=False)
    else:
        logger.info(f"read from file: {cache_file}")
        graph = nx.read_gpickle(cache_file)
        node_df = pd.read_csv(node_df_path)
        rel_df = pd.read_csv(rel_df_path)
    return graph, node_df, rel_df


def make_positive_batch(node: str, graph: nx.DiGraph) -> List[Dict]:
    ancestors = nx.descendants(graph, node)
    batch = [
        {"node": node, "pair_node": ancestor, "entail": 1}
        for ancestor in ancestors
    ]
    return batch


def make_negative_batch(
    node: str,
    graph: nx.DiGraph,
    all_nodes: Set,
    num_negative_entries: int = 10,
) -> List[Dict]:
    ancestors = set(nx.descendants(graph, node))
    non_ancestors = list(all_nodes.difference(ancestors))
    if len(non_ancestors) > 0:
        non_ancestors_candidates = random.sample(
            non_ancestors, k=min(num_negative_entries, len(non_ancestors))
        )
        batch = [
            {"node": node, "pair_node": pair, "entail": 0}
            for pair in non_ancestors_candidates
        ]
    else:
        batch = []
    return batch


def make_positive_sample(
    graph: nx.DiGraph,
    cache_file: Path,
    num_workers: int = cpu_count() - 2,
    overwrite: bool = False,
) -> pd.DataFrame:
    if not cache_file.exists() or overwrite:
        nodes = graph.nodes()
        chunksize = math.floor(len(nodes) / num_workers)
        efo_positive_sample = process_map(
            partial(make_positive_batch, graph=graph),
            nodes,
            max_workers=num_workers,
            chunksize=chunksize,
        )
        efo_positive_df = pd.DataFrame(
            [item for batch in efo_positive_sample for item in batch]
        )
        efo_positive_df.to_csv(cache_file, index=False)
    else:
        efo_positive_df = pd.read_csv(cache_file)
    logger.info(efo_positive_df.info())
    return efo_positive_df


def make_negative_sample(
    graph: nx.DiGraph,
    cache_file: Path,
    num_workers: int = cpu_count() - 2,
    num_negative_entries: int = 15,
    overwrite: bool = False,
) -> pd.DataFrame:
    if not cache_file.exists() or overwrite:
        nodes = graph.nodes()
        chunksize = math.floor(len(nodes) / num_workers)
        efo_negative_sample = process_map(
            partial(
                make_negative_batch,
                graph=graph,
                all_nodes=set(nodes),
                num_negative_entries=num_negative_entries,
            ),
            nodes,
            max_workers=num_workers,
            chunksize=chunksize,
        )
        efo_negative_df = pd.DataFrame(
            [item for batch in efo_negative_sample for item in batch]
        )
        efo_negative_df.to_csv(cache_file, index=False)
    else:
        efo_negative_df = pd.read_csv(cache_file)
    logger.info(efo_negative_df.info())
    return efo_negative_df


def make_combined_sample(
    efo_positive_sample: pd.DataFrame, efo_negative_sample: pd.DataFrame
) -> pd.DataFrame:
    # make balance category
    balance_sample_index = random.sample(
        range(len(efo_negative_sample)), k=len(efo_positive_sample)
    )
    negative_sample_balance = efo_negative_sample.iloc[balance_sample_index, :]
    combined_sample = pd.concat([efo_positive_sample, negative_sample_balance])
    # reshuffle
    combined_sample = combined_sample.iloc[
        random.sample(range(len(combined_sample)), k=len(combined_sample)), :,
    ]
    logger.info(combined_sample["entail"].value_counts())
    return combined_sample


def make_distance_batch(node: str, graph: nx.Graph, nodes_set: Set):
    target_nodes = list(nodes_set.difference(set([node])))
    distance = [
        {
            "source": node,
            "target": target_node,
            "distance": nx.shortest_path_length(graph, node, target_node),
        }
        for target_node in target_nodes
    ]
    df = pd.DataFrame(distance)
    return df


def make_distance_df_stage0(
    efo_graph: nx.DiGraph,
    cache_file: Path,
    num_workers: int = 4,
    overwrite: bool = False,
    load_to_runtime: bool = True,
) -> Optional[pd.DataFrame]:
    def _make():
        nodes = list(efo_graph.nodes())
        undirected_graph = efo_graph.to_undirected()
        # chunksize = math.floor(len(nodes) / num_workers)
        chunksize = 2
        distance_df_list = process_map(
            partial(
                make_distance_batch,
                graph=undirected_graph,
                nodes_set=set(nodes),
            ),
            nodes,
            max_workers=num_workers,
            chunksize=chunksize,
        )
        distance_df = pd.concat(distance_df_list)
        return distance_df

    if not cache_file.exists() or overwrite:
        distance_df = _make()
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"write to file: {cache_file}")
        distance_df.to_csv(cache_file, index=False)
        if load_to_runtime:
            return distance_df
    elif load_to_runtime:
        logger.info(f"read from file: {cache_file}")
        distance_df = pd.read_csv(cache_file)
        return distance_df
    return None


def get_efo_synonyms(
    node_label: str, efo_desc: pd.DataFrame, top_n: Optional[int] = None,
) -> Optional[List[str]]:
    """Get synonyms of an EFO node, when top_n, return only the
    first n items.
    """
    try:
        # [0] get the first elem from a indexed series slice
        res = efo_desc[efo_desc["label"] == node_label]["synonyms"][0]
        if top_n is not None and res is not None:
            res = res[:top_n]
    except:
        res = None
    return res


def make_distance_df_stage1(
    node_df: pd.DataFrame,
    efo_desc: pd.DataFrame,
    cache_file: Path,
    overwrite: bool = False,
):
    def _make_syn_df(label: str) -> Optional[pd.DataFrame]:
        synonyms = get_efo_synonyms(label, efo_desc)
        if isinstance(synonyms, list):
            res = pd.DataFrame(
                [
                    {"source": label, "target": _, "distance": 0}
                    for _ in synonyms
                ]
            )
            return res
        else:
            return None

    def _make() -> pd.DataFrame:
        self_dist = [_make_syn_df(_) for _ in node_df["label"]]
        self_dist_df = pd.concat(self_dist)
        return self_dist_df

    if not cache_file.exists() or overwrite:
        self_dist_df = _make()
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"write to file: {cache_file}")
        self_dist_df.to_csv(cache_file, index=False)
    else:
        logger.info(f"read from file: {cache_file}")
        self_dist_df = pd.read_csv(cache_file)
    return self_dist_df


def make_distance_db(
    df_list: List[pd.DataFrame],
    file_paths: Dict[str, Path],
    split_ratio: float = 0.75,
    overwrite: bool = False,
):
    def _make():
        logger.info("Doing train/val split.")
        # concat, then reshuffle
        combined_df = pd.concat(df_list).reset_index(drop=True)
        combined_len = len(combined_df)
        full_index = range(combined_len)
        train_index = random.sample(
            full_index, k=math.floor(split_ratio * combined_len)
        )
        val_index = list(set(full_index).difference(set(train_index)))
        train_df = combined_df.iloc[train_index, :].reset_index(drop=True)
        val_df = combined_df.iloc[val_index, :].reset_index(drop=True)
        meta_df = pd.DataFrame(
            [
                {"table": "TRAIN", "count": len(train_df)},
                {"table": "VALIDATION", "count": len(val_df)},
            ]
        )
        logger.info(f"Write to db {file_paths['train']}")
        with sqlite3.connect(file_paths["train"]) as conn:
            train_df.to_sql(
                "TRAIN",
                conn,
                index=True,
                index_label="idx",
                if_exists="replace",
            )
        logger.info(f"Write to db {file_paths['val']}")
        with sqlite3.connect(file_paths["val"]) as conn:
            val_df.to_sql(
                "VALIDATION",
                conn,
                index=True,
                index_label="idx",
                if_exists="replace",
            )
        logger.info(f"Write to csv {file_paths['meta']}")
        meta_df.to_csv(file_paths["meta"], index=False)

    files_exist = [val.exists() for key, val in file_paths.items()]
    if overwrite or not files_exist:
        logger.info("Regen files")
        _make()
    else:
        logger.info("Files exist. No changes.")
