import pandas as pd
from loguru import logger

from funcs.utils import find_project_root

ROOT = find_project_root()
# dependent data
UKBB_PATH = ROOT / "data" / "ukbb-test"
assert UKBB_PATH.exists()
PATH_TO_ORIG_PAIRS = UKBB_PATH / "ebi-ukb-cleaned.tsv"
assert PATH_TO_ORIG_PAIRS.exists()
PATH_TO_EFO = ROOT / "data" / "efo" / "epigraphdb_efo_nodes.csv"
assert PATH_TO_EFO.exists()
# task on trait efo match
INFERENCE_RESULTS_PATH = UKBB_PATH / "efo_bert_inference.csv"
assert INFERENCE_RESULTS_PATH.exists()
OUTPUT_PATH = UKBB_PATH / "bluebert_efo_mapping.csv"
OUTPUT_RANKING_PATH = UKBB_PATH / "bluebert_efo_rankings.csv"
# task on pairwise scores
INPUT_PAIRWISE_EFO_FULL_PATH = UKBB_PATH / "ukbb-efo-efo-pairs-full.csv"
INPUT_PAIRWISE_EFO_SCORES_PATH = (
    UKBB_PATH / "ukbb-efo-efo-pairs-text-scores.csv"
)
assert INPUT_PAIRWISE_EFO_FULL_PATH.exists()
assert INPUT_PAIRWISE_EFO_SCORES_PATH.exists()
OUTPUT_PAIRWISE_EFO_SCORES_PATH = UKBB_PATH / "ukbb-efo-efo-pairs-scores.csv"


def main():
    logger.info(f"Load from {INFERENCE_RESULTS_PATH}")
    df = pd.read_csv(INFERENCE_RESULTS_PATH)
    df = df.rename(columns={"text_2": "efo_term", "text_1": "ukbb_trait"})
    logger.info(
        f"""
    - len: {len(df)}
    - Num unique efo_terms: {len(df["efo_term"].drop_duplicates())}
    - Num unique ukbb_traits: {len(df["ukbb_trait"].drop_duplicates())}
    """
    )

    # top matching
    df_out = (
        df.groupby(by=["ukbb_trait"])
        .apply(lambda df: df.loc[df["score"].idxmin()])
        .reset_index(drop=True)
    )
    logger.info(f"""len: {len(df_out)}""")
    df_efo = pd.read_csv(PATH_TO_EFO).drop_duplicates(subset=["efo.value"])
    df_out = df_out.merge(
        df_efo[["efo.value", "efo.id"]].rename(
            columns={"efo.value": "efo_term", "efo.id": "efo_id"}
        ),
        left_on="efo_term",
        right_on="efo_term",
    )
    logger.info(f"""len: {len(df_out)}""")

    # ranking of expected term
    df_pairs = pd.read_csv(PATH_TO_ORIG_PAIRS, sep="\t")
    df1 = (
        df.groupby(by=["ukbb_trait"])
        .apply(
            lambda df: df.sort_values(by="score", ascending=True)
            .reset_index(drop=True)
            .assign(ranking=lambda df: df.index.tolist())
        )
        .merge(
            df_efo[["efo.value", "efo.id"]].rename(
                columns={"efo.value": "efo_term", "efo.id": "efo_id"}
            ),
            left_on="efo_term",
            right_on="efo_term",
        )
        .reset_index(drop=True)
    )
    df_ranking = df1.merge(
        df_pairs[["query", "full_id"]].rename(
            columns={"query": "ukbb_trait", "full_id": "efo_id"}
        ),
        left_on=["ukbb_trait", "efo_id"],
        right_on=["ukbb_trait", "efo_id"],
    )

    logger.info(f"write to {OUTPUT_PATH}")
    df_out.to_csv(OUTPUT_PATH, index=False)

    logger.info(f"write to {OUTPUT_RANKING_PATH}")
    df_ranking.to_csv(OUTPUT_RANKING_PATH, index=False)

    # pairwise efo
    efo_efo_full = pd.read_csv(INPUT_PAIRWISE_EFO_FULL_PATH)
    print(efo_efo_full.info())
    efo_efo_scores = pd.read_csv(INPUT_PAIRWISE_EFO_SCORES_PATH)
    print(efo_efo_scores.info())

    efo_efo_output = efo_efo_full.merge(
        efo_efo_scores.rename(
            columns={"text_1": "term_1", "text_2": "term_2"}
        ),
        left_on=["term_1", "term_2"],
        right_on=["term_1", "term_2"],
    )
    print(efo_efo_output.info())
    logger.info(f"write to {OUTPUT_PAIRWISE_EFO_SCORES_PATH}")
    efo_efo_output.to_csv(OUTPUT_PAIRWISE_EFO_SCORES_PATH, index=False)


if __name__ == "__main__":
    main()
