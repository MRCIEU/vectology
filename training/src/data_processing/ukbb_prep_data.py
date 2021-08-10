from itertools import combinations

import pandas as pd
from loguru import logger

from funcs.utils import find_project_root

ROOT = find_project_root()
PATH_TO_UKBB_DF = ROOT / "data" / "ukbb-test" / "ebi-ukb-cleaned.tsv"
assert PATH_TO_UKBB_DF.exists()
PATH_TO_EFO = ROOT / "data" / "efo" / "epigraphdb_efo_nodes.csv"
assert PATH_TO_EFO.exists()
PATH_TRAIT_EFO_OUTPUT = PATH_TO_UKBB_DF.parent / "ukbb-efo-pairs.csv"
PATH_TRAIT_TRAIT_OUTPUT = PATH_TO_UKBB_DF.parent / "ukbb-trait-trait-pairs.csv"
PATH_EFO_EFO_TEXT_OUTPUT = (
    PATH_TO_UKBB_DF.parent / "ukbb-efo-efo-pairs-text.csv"
)
PATH_EFO_EFO_FULL_OUTPUT = (
    PATH_TO_UKBB_DF.parent / "ukbb-efo-efo-pairs-full.csv"
)


def main():
    df_ukbb = pd.read_csv(PATH_TO_UKBB_DF, sep="\t")
    logger.info(df_ukbb.head())
    df_efo = pd.read_csv(PATH_TO_EFO)
    df_efo = df_efo.drop_duplicates(subset=["efo.value"])
    logger.info(df_efo.head())
    ukbb_trait_terms = df_ukbb["query"].drop_duplicates().tolist()
    efo_terms = df_efo["efo.value"].tolist()

    trait_efo_df = pd.concat(
        [
            pd.Series(efo_terms)
            .to_frame(name="efo_term")
            .assign(ukbb_trait=trait)
            for trait in ukbb_trait_terms
        ]
    ).reset_index(drop=True)
    trait_trait_df = pd.DataFrame.from_records(
        [
            {"text_1": _[0], "text_2": _[1]}
            for _ in combinations(ukbb_trait_terms, r=2)
        ]
    )

    efo_efo_id = pd.DataFrame.from_records(
        [
            {"id_1": _[0], "id_2": _[1]}
            for _ in combinations(
                df_ukbb["full_id"].drop_duplicates().tolist(), r=2
            )
        ]
    )
    efo_efo_full = efo_efo_id.merge(
        df_efo[["efo.value", "efo.id"]]
        .drop_duplicates()
        .rename(columns={"efo.value": "term_1", "efo.id": "id_1"}),
        left_on="id_1",
        right_on="id_1",
    ).merge(
        df_efo[["efo.value", "efo.id"]]
        .drop_duplicates()
        .rename(columns={"efo.value": "term_2", "efo.id": "id_2"}),
        left_on="id_2",
        right_on="id_2",
    )

    trait_efo_df = trait_efo_df.rename(
        columns={"efo_term": "text_2", "ukbb_trait": "text_1"}
    )
    logger.info(len(trait_efo_df))
    logger.info(len(ukbb_trait_terms) * len(efo_terms))
    logger.info(trait_efo_df)

    logger.info(f"Write to {PATH_TRAIT_EFO_OUTPUT}")
    trait_efo_df.to_csv(PATH_TRAIT_EFO_OUTPUT, index=False)

    logger.info(f"Write to {PATH_TRAIT_TRAIT_OUTPUT}")
    trait_trait_df.to_csv(PATH_TRAIT_TRAIT_OUTPUT, index=False)

    logger.info(f"Write to {PATH_EFO_EFO_FULL_OUTPUT}")
    efo_efo_full.to_csv(PATH_EFO_EFO_FULL_OUTPUT, index=False)

    logger.info(f"Write to {PATH_EFO_EFO_TEXT_OUTPUT}")
    efo_efo_full[["term_1", "term_2"]].rename(
        columns={"term_1": "text_1", "term_2": "text_2"}
    ).to_csv(PATH_EFO_EFO_TEXT_OUTPUT, index=False)


if __name__ == "__main__":
    main()
