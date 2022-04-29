import json
from typing import List, Optional

import pandas as pd
from icecream import ic
from metaflow import FlowSpec, step

from funcs import utils
from funcs.data_processing import stage1_processing
from funcs.nlp import nlp

proj_root = utils.find_project_root()
data_root = utils.find_data_root()

BIOCONCEPTVEC_MODEL_PATH = (
    proj_root / "models" / "bioconceptvec" / "bioconceptvec_word2vec_skipgram.bin"
)
assert BIOCONCEPTVEC_MODEL_PATH.exists()
PUBTATOR_DIR = data_root / "output" / "pubtator"
assert PUBTATOR_DIR.exists()


def read_session_res(session_id: str) -> Optional[str]:
    file_path = PUBTATOR_DIR / "batch_requests" / f"{session_id}.txt"
    if not file_path.exists():
        return None
    else:
        with file_path.open("r") as f:
            return f.read()


def verify_ner(session_res: Optional[str]) -> Optional[List[str]]:
    if session_res is None:
        return None
    raw_res = session_res.strip().split("\n")
    if len(raw_res) > 2:
        return raw_res[2:]
    else:
        return None


def ner_res_to_ent_id(ner_res: str) -> Optional[str]:
    ent_type_idx = 4
    ent_idx = 5
    expect_res_len = 6
    split = ner_res.split("\t")
    if len(split) < expect_res_len:
        return None
    if (len(split[ent_type_idx]) == 0) or (len(split[ent_idx]) == 0):
        return None
    res = "{ent_type}_{ent}".format(
        ent_type=split[ent_type_idx], ent=split[ent_idx].replace(":", "_")
    )
    return res


class PubtatorNerFlow(FlowSpec):
    @step
    def start(self):
        "start"
        self.ebi_df = stage1_processing.get_ebi_data()
        self.efo_node_df = stage1_processing.get_efo_nodes()
        ic(utils.df_info(self.ebi_df))
        ic(utils.df_info(self.efo_node_df))
        self.efo_nx = stage1_processing.get_efo_nx()
        self.next(self.make_terms_df)

    @step
    def make_terms_df(self):
        "Revise terms_df"
        ebi_terms = (
            self.ebi_df.assign(
                batch="ebi",
                term=lambda df: df["query"]
                .apply(nlp.ascii_fy)
                .apply(lambda x: None if x == "" else x),
            )[["batch", "mapping_id", "term"]]
            .dropna()
            .drop_duplicates(subset=["term"])
        )
        efo_terms = (
            (
                self.efo_node_df.assign(
                    batch="efo",
                    # NOTE: 1-based index
                    mapping_id=lambda df: pd.Series(range(len(df))) + 1,
                    term=lambda df: df["efo_label"]
                    .apply(nlp.ascii_fy)
                    .apply(lambda x: None if x == "" else x),
                )
            )[["batch", "mapping_id", "term"]]
            .dropna()
            .drop_duplicates(subset=["term"])
        )
        terms_df = (
            pd.concat([ebi_terms, efo_terms])
            .assign(mapping_id=lambda df: df["mapping_id"].astype(int))
            .reset_index(drop=True)
        )

        self.terms_df = terms_df
        self.next(self.make_ner_df)

    @step
    def make_ner_df(self):
        ner_res_path = PUBTATOR_DIR / "ner_res.json"
        if ner_res_path.exists():
            with ner_res_path.open() as f:
                self.ner_df = pd.DataFrame(json.load(f))
        else:
            self.ner_df = self._collect_pubtator_res()
            with ner_res_path.open("w") as f:
                json.dump(self.ner_df.to_dict(orient="records"), f)
        self.next(self.end)

    def _collect_pubtator_res(self):
        "Collect pubtator api results"
        session_df_path = PUBTATOR_DIR / "session_df_sanitized.csv"
        assert session_df_path.exists()
        batch_requests_dir = PUBTATOR_DIR / "batch_requests"
        assert batch_requests_dir.exists()

        session_df = pd.read_csv(session_df_path)
        ner_df = (
            self.terms_df.merge(session_df, on=["batch", "term"], how="inner")
            # this is matching by label, and found to contain merging dupes
            .drop_duplicates(subset=["batch", "term"]).reset_index(drop=True)
        )
        ner_df = ner_df.assign(
            exists=lambda df: df["session_id"].apply(
                lambda x: (batch_requests_dir / f"{x}.txt").exists()
            )
        ).pipe(utils.df_shape)
        ner_df = ner_df[ner_df["exists"]].reset_index(drop=True).pipe(utils.df_shape)
        ner_df = (
            ner_df.assign(
                session_res=lambda df: df["session_id"]
                .apply(read_session_res)
                .apply(verify_ner)
            )
            .dropna()
            .pipe(utils.df_shape)
            .assign(
                ner_res=lambda df: df["session_res"].apply(
                    lambda x_list: [ner_res_to_ent_id(_) for _ in x_list]
                )
                # .apply(
                #     lambda x_list: [
                #         _
                #         for _ in x_list
                #         if _ is not None and _ in self.bioconceptvec_keys
                #     ]
                # )
                .apply(lambda x_list: None if len(x_list) == 0 else x_list)
            )
            .dropna()
            .pipe(utils.df_shape)
            .reset_index(drop=True)
        )
        return ner_df

    @step
    def end(self):
        "end"

        ic(self.terms_df.info())
        ic(self.ner_df.info())


if __name__ == "__main__":
    PubtatorNerFlow()
