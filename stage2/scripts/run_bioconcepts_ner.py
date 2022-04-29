import json

import pandas as pd
import requests
from icecream import ic
from metaflow import FlowSpec, Parameter, step

from funcs import utils
from funcs.data_processing import stage1_processing
from funcs.nlp import nlp

import janitor  # noqa


INDEX_NAME = "bioconcepts"
ES_URL = "http://ieu-mrbssd1.epi.bris.ac.uk:26550"
DATA_ROOT = utils.find_data_root()

EXCLUDE_TERMS = ["and", "ands", "not", "with", "other"]


def search_term(term: str):
    search_data = {
        "query": {
            "bool": {
                "should": {
                    "match": {
                        "ent_term_norm": {
                            "query": term,
                        },
                    }
                },
                "must_not": [{"term": {"ent_term_norm": _}} for _ in EXCLUDE_TERMS],
            }
        },
        "aggs": {
            "dedup": {
                "terms": {"field": "ent_id"},
                "aggs": {"dedup_docs": {"top_hits": {"size": 1}}},
            }
        },
    }

    url = ES_URL + f"/{INDEX_NAME}" + "/_search"
    r = requests.get(url, json=search_data)
    r.raise_for_status()
    search_res = r.json()
    return search_res


class BioconceptNer(FlowSpec):

    OVERWRITE = Parameter(
        "overwrite",
        help="overwrite",
        default=False,
        is_flag=True,
    )

    @step
    def start(self):
        self.output_dir = DATA_ROOT / "output" / "bioconcepts"
        self.output_dir.mkdir(exist_ok=True)

        self.ebi_df = stage1_processing.get_ebi_data()
        self.efo_node_df = stage1_processing.get_efo_nodes()
        ic(utils.df_info(self.ebi_df))
        ic(utils.df_info(self.efo_node_df))

        self.next(self.make_terms_df)

    @step
    def make_terms_df(self):
        output_path = self.output_dir / "terms_df.csv"
        if not output_path.exists() or self.OVERWRITE:
            self.terms_df = self._make_terms_df()
            self.terms_df.to_csv(output_path, index=False)
        else:
            self.terms_df = pd.read_csv(output_path)
        self.next(self.make_search_res)

    @step
    def make_search_res(self):
        output_path = self.output_dir / "search_res.json"
        if not output_path.exists() or self.OVERWRITE:
            self.search_res = self._make_search_res()
            with output_path.open("w") as f:
                json.dump(self.search_res.to_dict(orient="records"), f)
        else:
            with output_path.open() as f:
                self.search_res = pd.DataFrame(json.load(f))
        self.next(self.collect_ner_res)

    @step
    def collect_ner_res(self):
        output_path = self.output_dir / "ner_res.json"
        if not output_path.exists() or self.OVERWRITE:
            self.ner_res = self._collect_ner_res()
            with output_path.open("w") as f:
                json.dump(self.ner_res.to_dict(orient="records"), f)
        else:
            with output_path.open() as f:
                self.ner_res = pd.DataFrame(json.load(f))
        self.next(self.end)

    @step
    def end(self):
        pass

    def _make_terms_df(self) -> pd.DataFrame:
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
        return terms_df

    def _make_search_res(self):
        ner_res = self.terms_df.assign(
            search_res=lambda df: df["term"].apply(search_term)
        )
        return ner_res

    def _collect_ner_res(self):
        def _pick(item):
            search_buckets = item["aggregations"]["dedup"]["buckets"]
            res = [
                {
                    "ent_id": _["key"],
                    "ent_term": _["dedup_docs"]["hits"]["hits"][0]["_source"][
                        "ent_term"
                    ],
                }
                for _ in search_buckets
            ]
            return res

        ner_res = self.search_res.assign(
            ner_res=lambda df: df["search_res"].apply(_pick)
        )[["batch", "mapping_id", "term", "ner_res"]]
        return ner_res


if __name__ == "__main__":
    BioconceptNer()
