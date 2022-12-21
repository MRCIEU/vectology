"""
stage 0 data prep
"""
import pandas as pd
from loguru import logger
from metaflow import FlowSpec, Parameter, step

from funcs import paths
from funcs.data_processing import stage1_processing

from icecream import ic  # noqa


STAGE2_CACHE = paths.stage2["output"]


def _make_ebi_df(ukb_master: pd.DataFrame, efo_node_df: pd.DataFrame) -> pd.DataFrame:
    # drop some columns
    ebi_df = ukb_master[
        ["ZOOMA QUERY", "MAPPED_TERM_LABEL", "MAPPED_TERM_URI", "MAPPING_TYPE"]
    ]
    ebi_df.rename(columns={"ZOOMA QUERY": "query"}, inplace=True)
    logger.info(f"\n{ebi_df.head()}")
    logger.info(ebi_df.shape)

    # drop rows with multiple mappings
    ebi_df = ebi_df[~ebi_df["MAPPED_TERM_URI"].str.contains(",", na=False)]
    logger.info(ebi_df.shape)
    ebi_df = ebi_df[
        # ~ebi_df["MAPPED_TERM_URI"].str.contains("|", na=False, regex=False)
        ~ebi_df["MAPPED_TERM_URI"].str.contains(r"\|", na=False)
    ]
    logger.info(ebi_df.shape)

    # clean up and lowercase
    ebi_df["id"] = ebi_df["MAPPED_TERM_URI"].str.strip()
    ebi_df["query"] = ebi_df["query"].str.lower()

    # remove underscores
    ebi_df["query"] = ebi_df["query"].str.replace("_", " ")

    # drop where query and id are duplicates
    ebi_df.drop_duplicates(subset=["query", "id"], inplace=True)
    logger.info(ebi_df.shape)

    # drop nan
    ebi_df.dropna(inplace=True)
    logger.info(ebi_df.shape)
    logger.info(ebi_df.head())

    # drop cases where query and matched text are identical
    logger.info(ebi_df.shape)
    ebi_df = ebi_df[
        ebi_df["query"].str.lower() != ebi_df["MAPPED_TERM_LABEL"].str.lower()
    ]
    logger.info(ebi_df.shape)

    # get counts of mapping type
    logger.info(f'\n{ebi_df["MAPPING_TYPE"].value_counts()}')

    # check data against efo nodes
    efo_node_ids = list(efo_node_df["efo_id"])
    logger.info(efo_node_ids[:5])
    ebi_ids = list(ebi_df["id"])
    logger.info(ebi_ids[:5])
    missing = []
    matched = []
    for i in ebi_ids:
        match = False
        for s in efo_node_ids:
            if i in s and match is False:
                matched.append(s)
                match = True
        if match is False:
            missing.append(i)
    logger.info(f"Missing: {len(missing)} {missing}")

    # remove missing from ukb data
    logger.info(f"\n{ebi_df.head()}")
    logger.info(ebi_df.shape)
    for i in missing:
        ebi_df = ebi_df.drop(ebi_df[ebi_df["id"].str.contains(i)].index)
    ebi_df["full_id"] = matched

    # add index as ID
    ebi_df["mapping_id"] = range(1, ebi_df.shape[0] + 1)
    logger.info(ebi_df.head())
    logger.info(ebi_df.shape)
    return ebi_df


class Stage0DataPrep(FlowSpec):

    OVERWRITE = Parameter(
        "overwrite",
        help="overwrite",
        default=False,
        is_flag=True,
    )

    VERBOSE = Parameter(
        "verbose",
        help="verbose",
        default=False,
        is_flag=True,
    )

    @step
    def start(self):
        self.next(self.make_ebi_df)

    @step
    def make_ebi_df(self):
        output_path = STAGE2_CACHE / "ebi_df.csv"
        if not output_path.exists() or self.OVERWRITE:
            print("(re-)make ebi_df")
            raw_file_path = paths.init["ukb_master"]
            assert raw_file_path.exists(), raw_file_path

            raw_df = pd.read_csv(raw_file_path, sep="\t")
            efo_node_df = stage1_processing.get_efo_nodes()
            ebi_df = _make_ebi_df(ukb_master=raw_df, efo_node_df=efo_node_df)
            ebi_df.to_csv(output_path, index=False)
        else:
            print("skip")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    Stage0DataPrep()
