import time
from string import punctuation
from typing import Optional

import pandas as pd
import requests
from loguru import logger
from pydash import py_

from funcs import utils
from funcs.data_processing import stage1_processing
from funcs.utils import ic

ANNOTATE_URL_TEMPLATE = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/submit/{concept_type}"
SESSION_URL_TEMPLATE = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/retrieve/{session_id}"

data_root = utils.find_data_root()
assert data_root.exists()
OUTPUT_DIR = data_root / "output" / "pubtator"
BATCH_REQUEST_DIR = OUTPUT_DIR / "batch_requests"
BATCH_REQUEST_DIR.mkdir(exist_ok=True, parents=True)

ECHO_STEP = 200
CHUNK_SIZE = 3
CHUNK_SLEEP = 1


def ascii_fy(text: str) -> str:
    text = text.encode("ascii", "ignore").decode()
    for _ in punctuation:
        text = text.replace(_, " ")
    return text


def request_session_id(term: str) -> Optional[str]:
    url = ANNOTATE_URL_TEMPLATE.format(concept_type="All")
    try:
        r = requests.post(url, data=term.encode("utf-8"))
    except:
        return None
    return r.text


def request_session(terms_df: pd.DataFrame) -> pd.DataFrame:
    batch_terms = [
        {
            "batch": _["batch"],
            "term": _["term"],
            "session_id": None,
        }
        for _ in terms_df.to_dict(orient="records")
    ]
    batch_term_chunks = py_.chunk(batch_terms, size=CHUNK_SIZE)
    for chunk_idx, chunk in enumerate(batch_term_chunks):
        if chunk_idx % ECHO_STEP == 0:
            logger.info(f"#{chunk_idx} / {len(batch_term_chunks)}")
        for idx, elem in enumerate(chunk):
            term = elem["term"]
            session_id = request_session_id(term=term)
            batch_term_chunks[chunk_idx][idx]["session_id"] = session_id
        time.sleep(CHUNK_SLEEP)
    flat_terms = py_.flatten(batch_term_chunks)
    ic(flat_terms[:5])
    res = pd.DataFrame(flat_terms)
    return res


def make_terms_df() -> pd.DataFrame:
    ebi_data = stage1_processing.get_ebi_data()
    efo_nodes = stage1_processing.get_efo_nodes()
    ebi_terms = (
        ebi_data["query"]
        .apply(ascii_fy)
        .apply(lambda x: None if x == "" else x)
        .dropna()
        .tolist()
    )
    efo_terms = (
        efo_nodes["efo_label"]
        .apply(ascii_fy)
        .apply(lambda x: None if x == "" else x)
        .dropna()
        .tolist()
    )
    terms_df = pd.concat(
        [
            pd.DataFrame([{"batch": "ebi", "term": _} for _ in ebi_terms]),
            pd.DataFrame([{"batch": "efo", "term": _} for _ in efo_terms]),
        ]
    ).reset_index(drop=True)
    return terms_df


def main():
    terms_df_path = OUTPUT_DIR / "terms_df.csv"
    if terms_df_path.exists():
        terms_df = pd.read_csv(terms_df_path)
    else:
        terms_df = make_terms_df()
        terms_df.to_csv(terms_df_path, index=False)
    ic(utils.df_info(terms_df))

    session_df = request_session(terms_df=terms_df)
    ic(utils.df_info(session_df))
    session_df_path = OUTPUT_DIR / "session_df.csv"
    session_df.to_csv(session_df_path, index=False)


if __name__ == "__main__":
    main()
