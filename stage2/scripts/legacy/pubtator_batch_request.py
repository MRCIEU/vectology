import json
import time
from string import punctuation
from typing import Any, Dict, List

import requests
from pydash import py_
from nltk import word_tokenize
from nltk.corpus import stopwords

from funcs import utils
from funcs.data_processing import stage1_processing
from funcs.utils import ic

ANNOTATE_URL_TEMPLATE = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/submit/{concept_type}"

data_root = utils.find_data_root()
assert data_root.exists()
OUTPUT_DIR = data_root / "output" / "pubtator"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
BATCH_SIZE = 1_000
SLEEP_SECS = 20


def sanitize(text_list: List[str]) -> List[str]:
    def _proc(text: str) -> str:
        text = text.encode("ascii", "ignore").decode()
        for _ in punctuation:
            text = text.replace(_, " ")
        # tokens = [
        #     token
        #     for token in word_tokenize(text)
        #     if token not in punctuation and token not in stop_words
        # ]
        return text

    stop_words = set(stopwords.words("english"))
    cleaned = [_proc(_) for _ in text_list]
    return cleaned


def make_batch_request(terms: List[str]) -> List[Dict[str, Any]]:
    terms_chunked = py_.chunk(terms, size=BATCH_SIZE)
    url = ANNOTATE_URL_TEMPLATE.format(concept_type="All")
    batch_info_col = []
    for idx, chunk in enumerate(terms_chunked):
        ic(f"#{idx} / {len(terms_chunked)}")
        # [start_idx, end_idx)
        start_idx = idx * BATCH_SIZE
        end_idx = start_idx + len(chunk)
        sanitized_chunk = sanitize(chunk)
        terms_concat = "\n-*-\n".join(sanitized_chunk)
        r = requests.post(url, data=terms_concat.encode("utf-8"))
        try:
            r.raise_for_status()
            session_id = r.text
            ic(f"{idx}: {session_id}")
            batch_info = {
                "idx": idx,
                "session_id": session_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "terms": sanitized_chunk,
            }
            batch_info_col.append(batch_info)
        except:
            ic(f"Error: {idx}; {sanitized_chunk}")
        time.sleep(SLEEP_SECS)
    return batch_info_col


def main():
    ebi_data = stage1_processing.get_ebi_data()
    efo_nodes = stage1_processing.get_efo_nodes()
    ebi_terms = ebi_data["query"].tolist()
    efo_terms = efo_nodes["efo_label"].tolist()

    batch_info_list = []
    for batch, terms in [("ebi", ebi_terms), ("efo", efo_terms)]:
        ic(batch)
        info_list = make_batch_request(terms=terms)
        batch_request_info = [dict({"batch": batch}, **_) for _ in info_list]
        batch_info_list = batch_info_list + batch_request_info

    batch_info_path = OUTPUT_DIR / "batch_request_info.json"
    with batch_info_path.open("w") as f:
        json.dump(batch_info_list, f)


if __name__ == "__main__":
    main()
