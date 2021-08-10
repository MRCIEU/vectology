import argparse
import random

import pandas as pd
from loguru import logger

import settings
from funcs.efo.efo_data_processing import (
    make_combined_sample,
    make_distance_db,
    make_distance_df_stage0,
    make_distance_df_stage1,
    make_efo_gwas_graph,
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
OUTPUT_DIR = ROOT / "output" / "efo-gwas"
GRAPH_FILE = OUTPUT_DIR / "efo_gwas_graph.gpickle"
POSITIVE_SAMPLE_FULL = OUTPUT_DIR / "positive_full.csv"
NEGATIVE_SAMPLE_FULL = OUTPUT_DIR / "negative_full.csv"
COMBINED_SAMPLE = OUTPUT_DIR / "combined.csv"
DISTANCE_DF_STAGE0 = OUTPUT_DIR / "distance_stage0.csv"
DISTANCE_DF_STAGE1 = OUTPUT_DIR / "distance_stage1.csv"
DISTANCE_META = OUTPUT_DIR / "distance_meta.csv"
DISTANCE_DB_TRAIN = OUTPUT_DIR / "distance_train.db"
DISTANCE_DB_VAL = OUTPUT_DIR / "distance_val.db"
# dependency
DEPENDENCY = {
    "input": [
        EFO_DATA_DIR,
        EFO_NODE_FILE,
        GWAS_CATALOG_DIR,
        GWAS_CATALOG_FILE,
    ],
    "output": [
        OUTPUT_DIR,
        GRAPH_FILE,
        POSITIVE_SAMPLE_FULL,
        NEGATIVE_SAMPLE_FULL,
        COMBINED_SAMPLE,
        DISTANCE_DF_STAGE0,
        DISTANCE_DF_STAGE1,
        DISTANCE_META,
        DISTANCE_DB_TRAIN,
        DISTANCE_DB_VAL,
    ],
}
# hyperparams
# number of negative entries for a node
NUM_NEGATIVE_ENTRIES = 20
SEED = settings.SEED


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite cache"
    )
    parser.add_argument(
        "-j", "--num_workers", default=settings.NUM_WORKERS, type=int
    )
    return parser


def main():
    random.seed(SEED)
    parser = create_parser()
    args = parser.parse_args()
    logger.info(f"args: {args}")

    check_dependent_files(DEPENDENCY)

    efo_node = pd.read_csv(EFO_NODE_FILE)
    efo_rel = pd.read_csv(EFO_REL_FILE)
    gwas_df = pd.read_csv(GWAS_CATALOG_FILE, sep="\t")
    efo_desc = pd.read_json(EFO_DETAILS_FILE, orient="index").rename_axis("id")
    logger.info("Make efo_graph")
    efo_graph, node_df, rel_df = make_efo_gwas_graph(
        efo_node=efo_node,
        efo_rel=efo_rel,
        efo_desc=efo_desc,
        gwas_df=gwas_df,
        cache_file=GRAPH_FILE,
        overwrite=args.overwrite,
    )

    # positive sample
    logger.info("Make positive sample")
    efo_positive_full = make_positive_sample(
        efo_graph,
        POSITIVE_SAMPLE_FULL,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    # negative sample
    logger.info("Make negative sample")
    efo_negative_full = make_negative_sample(
        efo_graph,
        NEGATIVE_SAMPLE_FULL,
        num_negative_entries=NUM_NEGATIVE_ENTRIES,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    assert len(efo_negative_full) >= len(efo_positive_full)
    # combined sample
    if not COMBINED_SAMPLE.exists() or args.overwrite:
        efo_combined_sample = make_combined_sample(
            efo_positive_full, efo_negative_full
        )
        efo_combined_sample.to_csv(COMBINED_SAMPLE, index=False)

    # distance df, stage 0: pairwise
    distance_df_stage0 = make_distance_df_stage0(
        efo_graph,
        cache_file=DISTANCE_DF_STAGE0,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )

    # distance df, stage 1: self distance with synonyms
    distance_df_stage1 = make_distance_df_stage1(
        node_df=node_df[node_df["source"] == "efo"],
        efo_desc=efo_desc,
        cache_file=DISTANCE_DF_STAGE1,
        overwrite=args.overwrite,
    )

    # write to db
    make_distance_db(
        df_list=[distance_df_stage0, distance_df_stage1],
        file_paths={
            "meta": DISTANCE_META,
            "train": DISTANCE_DB_TRAIN,
            "val": DISTANCE_DB_VAL,
        },
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
