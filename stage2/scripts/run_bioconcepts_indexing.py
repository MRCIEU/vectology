import json
import re
from typing import List

import pandas as pd
import requests
from metaflow import FlowSpec, Parameter, step
from pydash import py_

from funcs import utils

import janitor  # noqa


INDEX_NAME = "bioconcepts"
ES_URL = "http://ieu-mrbssd1.epi.bris.ac.uk:26550"
DATA_ROOT = utils.find_data_root()
BIOCONCEPTVEC_DIR = DATA_ROOT / "bioconceptvec"
assert BIOCONCEPTVEC_DIR.exists(), BIOCONCEPTVEC_DIR

INDEX_CONFIG_DATA = {
    "settings": {
        "analysis": {
            "analyzer": {
                "substring": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "kstem", "substring"],
                },
                "exact": {
                    "type": "custom",
                    "tokenizer": "keyword",
                    "filter": [
                        "lowercase",
                        "kstem",
                    ],
                },
            },
            "filter": {"substring": {"type": "shingle", "output_unigrams": True}},
        }
    },
    "mappings": {
        "properties": {
            "ent_id": {
                "type": "keyword",
            },
            "ent_term": {
                "type": "text",
            },
            "ent_term_norm": {
                "type": "text",
                "analyzer": "exact",
                "search_analyzer": "substring",
            },
        }
    },
}


def valid_p(term: str) -> bool:
    """
    - term should be long enough
    - term should not contain digit
    """
    num_char_limit = 3
    if len(term) <= num_char_limit:
        return False
    find_digits = re.findall(r"\d", term)
    if len(find_digits) > 0:
        return False
    return True


def clean_term(term: str) -> List[str]:
    # split by "|"
    terms = term.split("|")
    terms = [_.strip().lower() for _ in terms]
    # drop invalid terms
    terms = [_ for _ in terms if valid_p(_)]
    terms = py_.chain(terms).uniq().value()
    return terms


def index_chunk(index_name, docs):
    index_url = ES_URL + f"/{index_name}" + "/_bulk"
    headers = {"Content-Type": "application/x-ndjson", "charset": "UTF-8"}
    delim = {"index": {}}
    arr = []
    for _ in docs:
        arr.append(delim)
        arr.append(_)
    payload = "\n".join([json.dumps(_) for _ in arr]) + "\n"
    r = requests.post(index_url, data=payload, headers=headers)
    r.raise_for_status()
    return True


def bulk_index(index_name, docs, chunksize=500, log_step=20):
    docs_chunks = py_.chunk(docs, size=chunksize)
    print(f"split {len(docs)} into {len(docs_chunks)} chunks")
    for idx, chunk in enumerate(docs_chunks):
        if idx % log_step == 0:
            print(f"# chunk {idx}")
        index_chunk(index_name, chunk)


class BioconceptIndex(FlowSpec):

    OVERWRITE = Parameter(
        "overwrite",
        help="overwrite",
        default=False,
        is_flag=True,
    )

    @step
    def start(self):
        "Init."

        # self.input_data_path = BIOCONCEPTVEC_DIR / "bioconcepts2pubtatorcentral"
        self.input_data_path = BIOCONCEPTVEC_DIR / "disease2pubtatorcentral"
        assert self.input_data_path.exists(), self.input_data_path

        self.output_dir = DATA_ROOT / "output" / "bioconcepts"
        self.output_dir.mkdir(exist_ok=True)
        self.chunksize = 500_000

        r = requests.get(ES_URL)
        assert r.ok, ES_URL
        self.next(self.init_index)

    @step
    def init_index(self):
        url = ES_URL + f"/{INDEX_NAME}"
        r = requests.get(url)
        if r.ok:
            print("Remove index")
            r = requests.delete(url)

        print("Init index")
        r = requests.put(url, json=INDEX_CONFIG_DATA)
        r.raise_for_status()
        self.next(self.index_vocab)

    @step
    def index_vocab(self):
        idx = 0
        with pd.read_csv(
            self.input_data_path,
            sep="\t",
            names=["idx", "ent_type", "ent_id", "term", "source"],
            chunksize=self.chunksize,
        ) as reader:
            for chunk in reader:
                print(f"chunk #{idx}")
                idx = idx + 1
                print("  load data")
                index_data = (
                    chunk.dropna()
                    .transform_column("term", clean_term, "expand_terms")
                    .explode("expand_terms")
                    .dropna()
                    .drop_duplicates(subset=["ent_id", "expand_terms"])
                    .drop_duplicates(subset=["expand_terms"])[
                        ["ent_type", "ent_id", "expand_terms"]
                    ]
                    .reset_index(drop=True)
                    .rename(columns={"expand_terms": "ent_term"})
                    .assign(
                        ent_id=lambda df: df.apply(
                            lambda row: row["ent_type"]
                            + "_"
                            + row["ent_id"].replace(":", "_"),
                            axis=1,
                        ),
                        ent_term_norm=lambda df: df["ent_term"],
                    )[["ent_id", "ent_term", "ent_term_norm"]]
                    .to_dict(orient="records")
                )
                print("  index data")
                bulk_index(index_name=INDEX_NAME, docs=index_data)
        self.next(self.end)

    @step
    def end(self):
        "Done."


if __name__ == "__main__":
    BioconceptIndex()
