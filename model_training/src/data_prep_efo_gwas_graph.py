"""Preprocess efo data
"""
import json

import pandas as pd
from loguru import logger

from funcs import data_cleaning
from funcs.utils import find_project_root

ROOT = find_project_root()
EFO_DATA_DIR = ROOT / "data" / "efo-2021-05-24"
# input
EFO_NODE_FILE = EFO_DATA_DIR / "efo_nodes_2021-05-24.csv"
EFO_REL_FILE = EFO_DATA_DIR / "efo_edges_2021-05-24.csv"
EFO_DETAIL_FILE = EFO_DATA_DIR / "efo-v3.29.1.json"
INPUT_FILES = [EFO_DETAIL_FILE, EFO_NODE_FILE, EFO_REL_FILE]
# output
OUTPUT_DIR = ROOT / "data" / "cleaned_data"
EFO_GRAPH_FILE = OUTPUT_DIR / "efo_gwas_graph.gpickle"

def main():
    # init
    [print(f"{_}:{_.exists()}") for _ in INPUT_FILES]
    OUTPUT_DIR.mkdir(exist_ok=True)
    efo_node = pd.read_csv(EFO_NODE_FILE)
    efo_rel = pd.read_csv(EFO_REL_FILE)

    # cleaning
    efo_node_clean = efo_node
    efo_rel_clean = efo.clean_efo_rel(efo_rel)

    pass

if __name__ == "__main__":
    main()
