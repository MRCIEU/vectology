import random

import pandas as pd
from loguru import logger

import settings
from funcs.efo.efo_data_processing import (
    make_combined_sample,
    make_efo_graph,
    make_negative_sample,
    make_positive_sample,
)
from funcs.utils import check_dependent_files, find_project_root

ROOT = find_project_root()
# efo data
EFO_DATA_DIR = ROOT / "data" / "efo"
EFO_NODE_FILE = EFO_DATA_DIR / "epigraphdb_efo_nodes.csv"
EFO_REL_FILE = EFO_DATA_DIR / "epigraphdb_efo_rels.csv"
EFO_DETAILS_FILE = EFO_DATA_DIR / "efo_details_simplified.json"
# gwas catalog data
GWAS_CATALOG_DIR = ROOT / "data" / "gwas-catalog"
GWAS_CATALOG_FILE = (
    GWAS_CATALOG_DIR / "gwas_catalog_v1.0.2-studies_r2020-11-03.tsv"
)
# output
EFO_OUTPUT_DIR = ROOT / "output" / "efo"
EFO_GRAPH_FILE = EFO_OUTPUT_DIR / "efo_graph.gpickle"
EFO_POSITIVE_SAMPLE_FULL = EFO_OUTPUT_DIR / "efo_positive_full.csv"
EFO_NEGATIVE_SAMPLE_FULL = EFO_OUTPUT_DIR / "efo_negative_full.csv"
EFO_COMBINED_SAMPLE = EFO_OUTPUT_DIR / "efo_combined.csv"
# dependency
DEPENDENCY = {
    "input": [
        EFO_DATA_DIR,
        EFO_NODE_FILE,
        GWAS_CATALOG_DIR,
        GWAS_CATALOG_FILE,
    ],
    "output": [
        EFO_OUTPUT_DIR,
        EFO_GRAPH_FILE,
        EFO_POSITIVE_SAMPLE_FULL,
        EFO_NEGATIVE_SAMPLE_FULL,
        EFO_COMBINED_SAMPLE,
    ],
}
# hyperparams
# number of negative entries for a node
NUM_NEGATIVE_ENTRIES = 20
SEED = settings.SEED


def main():
    random.seed(SEED)
    check_dependent_files(DEPENDENCY)
    efo_node = pd.read_csv(EFO_NODE_FILE)
    efo_rel = pd.read_csv(EFO_REL_FILE)
    # efo_desc = pd.read_json(EFO_DETAILS_FILE, orient="index").rename_axis("id")
    logger.info("Make efo_graph")
    efo_graph = make_efo_graph(efo_node, efo_rel, EFO_GRAPH_FILE)

    # positive sample
    logger.info("Make positive sample")
    efo_positive_full = make_positive_sample(
        efo_graph, EFO_POSITIVE_SAMPLE_FULL
    )
    # negative sample
    logger.info("Make negative sample")
    efo_negative_full = make_negative_sample(
        efo_graph,
        EFO_NEGATIVE_SAMPLE_FULL,
        num_negative_entries=NUM_NEGATIVE_ENTRIES,
    )
    assert len(efo_negative_full) >= len(efo_positive_full)
    # combined sample
    efo_combined_sample = make_combined_sample(
        efo_positive_full, efo_negative_full
    )
    efo_combined_sample.to_csv(EFO_COMBINED_SAMPLE, index=False)


if __name__ == "__main__":
    main()
