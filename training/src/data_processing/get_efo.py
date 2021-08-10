import argparse
import json
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd
import requests
from loguru import logger

from funcs.utils import find_project_root

EPIGRAPHDB_API_URL = "https://api.epigraphdb.org"
EBI_OLS_API_URL = "http://www.ebi.ac.uk/ols/api/ontologies"
BATCH_SIZE = 1_200
# files
DATA_DIR = find_project_root() / "data/efo"
EFO_NODES = DATA_DIR / "epigraphdb_efo_nodes.csv"
EFO_RELS = DATA_DIR / "epigraphdb_efo_rels.csv"
EFO_DETAILS = DATA_DIR / "efo_details.json"
EFO_DESC = DATA_DIR / "efo_details_simplified.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
dependency = {
    "input": [],
    "output": [EFO_NODES, EFO_RELS, EFO_DETAILS, EFO_DESC]
}


def main(args: argparse.Namespace) -> None:

    # params {
    n_procs = min(args.n_procs, cpu_count() - 1)
    overwrite = args.overwrite
    params = f"""
    - BATCH_SIZE: {BATCH_SIZE}
    - DATA_DIR: {DATA_DIR}
    - EFO_NODES: {EFO_NODES}; {EFO_NODES.exists()}
    - EFO_RELS: {EFO_RELS}; {EFO_RELS.exists()}
    - EFO_DETAILS: {EFO_DETAILS}; {EFO_DETAILS.exists()}
    - EFO_DESC: {EFO_DESC}; {EFO_DESC.exists()}
    - overwrite: {overwrite}
    - n_procs: {n_procs}
    """
    logger.info(f"params: {params}")
    # }

    # get efo nodes {
    if overwrite or not EFO_NODES.exists():
        # get size of entities
        n_nodes = get_node_size()
        logger.info(f"n_nodes: {n_nodes:_}")
        efo_nodes_df = (
            pd.concat(
                [
                    get_efo_nodes(skip=skip, limit=BATCH_SIZE)
                    for skip in range(0, n_nodes + 1, BATCH_SIZE)
                ]
            )
            .drop_duplicates()
            .reset_index(drop=True)
            # .assign(ontology=lambda df: df["efo.id"].apply(get_ontology))
        )
        # save
        efo_nodes_df.to_csv(EFO_NODES, index=False)
    else:
        efo_nodes_df = pd.read_csv(EFO_NODES)
    logger.info(f"\n{efo_nodes_df.info()}\n{efo_nodes_df.head()}")
    # }

    # get efo rels {
    if overwrite or not EFO_RELS.exists():
        n_rels = get_rel_size()
        logger.info(f"n_rels: {n_rels}")
        efo_rels_df = (
            pd.concat(
                [
                    get_efo_rels(skip=skip, limit=BATCH_SIZE)
                    for skip in range(0, n_rels + 1, BATCH_SIZE)
                ]
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )
        efo_rels_df.to_csv(EFO_RELS, index=False)
    else:
        efo_rels_df = pd.read_csv(EFO_RELS)
    logger.info(f"\n{efo_rels_df.info()}\n{efo_rels_df.head()}")
    # }

    # get efo detail data {
    if overwrite or not EFO_DETAILS.exists():
        uri_list = efo_nodes_df["efo.id"].tolist()
        with Pool(n_procs) as pool:
            efo_detail = pool.map(
                count_wrap_get_efo_detail,
                [(i, uri) for i, uri in enumerate(uri_list)],
            )
        with EFO_DETAILS.open("w") as f:
            json.dump(efo_detail, f)
    # }

    # get efo descriptions, simplified from the details json {
    if overwrite or not EFO_DESC.exists():
        with EFO_DETAILS.open() as f:
            efo_detail = json.load(f)
        desc = [get_efo_desc(item) for item in efo_detail]
        desc = [
            _ for _ in desc if _ is not None
        ]  # remove entries without value
        efo_desc_df = pd.DataFrame.from_records(
            [_ for _ in desc if _ is not None]
        )
        efo_desc_df.set_index("id").to_json(EFO_DESC, orient="index")
    else:
        efo_desc_df = pd.read_json(EFO_DESC, orient="index").rename_axis("id")
    logger.info(f"\n{efo_desc_df.info()}\n{efo_desc_df.head()}")
    # }
    return None


def get_node_size() -> int:
    query = """
    MATCH (efo:Efo)
    RETURN size(collect(DISTINCT efo)) AS n_nodes
    """
    r = requests.post(f"{EPIGRAPHDB_API_URL}/cypher", json={"query": query})
    r.raise_for_status()
    res = r.json()["results"]
    return res[0]["n_nodes"]


def get_rel_size() -> int:
    query = """
    MATCH (efo:Efo)-[r:EFO_CHILD_OF]->(parent_efo:Efo)
    RETURN size(collect(DISTINCT r)) AS n_rels
    """
    r = requests.post(f"{EPIGRAPHDB_API_URL}/cypher", json={"query": query})
    r.raise_for_status()
    res = r.json()["results"]
    return res[0]["n_rels"]


def get_efo_nodes(skip: int, limit: int) -> pd.DataFrame:
    logger.info(f"EFO nodes, skip: {skip:_}")
    query = f"""
    MATCH (efo:Efo)
    RETURN efo
    SKIP {skip}
    LIMIT {limit}
    """
    r = requests.post(f"{EPIGRAPHDB_API_URL}/cypher", json={"query": query})
    r.raise_for_status()
    res = pd.json_normalize(r.json()["results"])
    return res


def get_efo_rels(skip: int, limit: int) -> pd.DataFrame:
    logger.info(f"EFO rels, skip: {skip:_}")
    query = f"""
    MATCH (efo:Efo)-[r:EFO_CHILD_OF]->(parent_efo:Efo)
    RETURN
        efo {{.id}},
        parent_efo {{.id}}
    SKIP {skip}
    LIMIT {limit}
    """
    r = requests.post(f"{EPIGRAPHDB_API_URL}/cypher", json={"query": query})
    r.raise_for_status()
    res = pd.json_normalize(r.json()["results"])
    return res


def count_wrap_get_efo_detail(input: Tuple[int, str]):
    i: int = input[0]
    uri: str = input[1]
    if i % 500 == 0:
        logger.info(f"efo detail: {i}")
    res = get_efo_detail_by_uri(uri)
    return res


def get_efo_detail_by_uri(uri: str) -> Dict[str, Any]:
    # double quote
    uri_quote = quote_plus(quote_plus(uri))
    ontology = "efo"
    url = f"{EBI_OLS_API_URL}/{ontology}/terms/{uri_quote}"
    r = requests.get(url)
    try:
        r.raise_for_status()
        detail = r.json()
    except:
        detail = None
    res = {uri: detail}
    return res


def get_efo_desc(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # item is structured as {{"foo": {...}}}
    item = list(item.values())[0]
    if item is not None:
        keys = item.keys()
        id = item["iri"] if "iri" in keys else None
        ontology = item["ontology_name"] if "ontology_name" in keys else None
        if id is not None:
            # desc
            desc = None
            if "description" in keys:
                desc = item["description"]
                if isinstance(desc, list):
                    desc = desc[0]
            # label
            label = item["label"] if "label" in keys else None
            # synonyms
            synonyms = item["synonyms"] if "synonyms" in keys else None
            res = {
                "id": id,
                "ontology": ontology,
                "label": label,
                "description": desc,
                "synonyms": synonyms,
            }
            return res
        else:
            return None
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite cache"
    )
    parser.add_argument(
        "-j",
        "--n-procs",
        default=8,
        type=int,
        help="Num of cpu cores for multiporcessing",
    )
    args = parser.parse_args()
    main(args=args)
