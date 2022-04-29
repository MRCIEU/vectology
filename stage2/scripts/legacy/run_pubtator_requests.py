import json
import time
from string import punctuation
from typing import Any, Dict, List

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

CHUNK_SIZE = 200
SUPER_CHUNK_SIZE = 2
MAX_NUM_TRIES = 5
INIT_TRY_INTERVAL = 15 * 60
TRY_INTERVAL = 10 * 60


def sanitize(text_list: List[str]) -> List[str]:
    def _proc(text: str) -> str:
        text = text.encode("ascii", "ignore").decode()
        for _ in punctuation:
            text = text.replace(_, " ")
        return text

    cleaned = [_proc(_) for _ in text_list]
    return cleaned


def request_session(terms: List[str]) -> str:
    url = ANNOTATE_URL_TEMPLATE.format(concept_type="All")
    terms_concat = "\n-*-\n".join(terms)
    r = requests.post(url, data=terms_concat.encode("utf-8"))
    r.raise_for_status()
    session_id = r.text
    return session_id


def retrieve_session(session_id) -> str:
    url = SESSION_URL_TEMPLATE.format(session_id=session_id)
    r = requests.get(url)
    r.raise_for_status()
    return r.text


def make_batch_request(chunks: List[Dict[str, Any]]) -> None:
    request_sessions = [
        {
            "batch": _["batch"],
            "idx": _["idx"],
            "session_id": None,
            "state": False,
            "output_path": BATCH_REQUEST_DIR
            / "{batch}-{idx}.txt".format(batch=_["batch"], idx=_["idx"]),
        }
        for _ in chunks
    ]
    logger.info(
        "Batches: {idx_list}".format(idx_list=[_["idx"] for _ in request_sessions])
    )
    # if all idx in the chunks have had output files exist
    all_done = sum([_["output_path"].exists() for _ in request_sessions]) == len(
        request_sessions
    )
    if all_done:
        logger.info("All files exists, skip")
        return None

    # submit annotation request
    logger.info("submit requests")
    for idx, chunk in enumerate(chunks):
        if request_sessions[idx]["output_path"].exists():
            continue
        session_id = request_session(terms=chunk["terms"])
        request_sessions[idx]["session_id"] = session_id
        request_sessions[idx]["state"] = False
    ic(request_sessions)
    time.sleep(INIT_TRY_INTERVAL)
    # retrieve
    logger.info("retrieve requests")
    for session_info in request_sessions:
        for nth_try in range(MAX_NUM_TRIES):
            # already retrieved
            if session_info["state"] or session_info["output_path"].exists():
                continue
            logger.info(
                f"{session_info['batch']}-{session_info['idx']}: " + f"#{nth_try} try"
            )
            try:
                session_results = retrieve_session(session_info["session_id"])
            except Exception as e:
                print(e)
                logger.info(
                    f"{session_info['batch']}-{session_info['idx']}: "
                    + f"#{nth_try} try: failed"
                )
                time.sleep(TRY_INTERVAL)
                continue
            session_info["state"] = True
            logger.info(
                f"{session_info['batch']}-{session_info['idx']}: "
                + f"#{nth_try} try: success"
            )
            with session_info["output_path"].open("w") as f:
                f.write(session_results)


def main():
    ebi_data = stage1_processing.get_ebi_data()
    efo_nodes = stage1_processing.get_efo_nodes()
    ebi_terms = sanitize(ebi_data["query"].tolist())
    efo_terms = sanitize(efo_nodes["efo_label"].tolist())

    logger.info("Preparing batch_request_info")
    batch_info_list = []
    for batch, terms in [("ebi", ebi_terms), ("efo", efo_terms)]:
        ic(batch)
        term_chunks = py_.chunk(terms, size=CHUNK_SIZE)
        term_chunks_annotated = [
            {
                "batch": batch,
                "idx": idx,
                # [start_idx, end_idx)
                "start_idx": idx * CHUNK_SIZE,
                "end_idx": (idx * CHUNK_SIZE) + len(_),
                "terms": _,
            }
            for idx, _ in enumerate(term_chunks)
        ]
        batch_info_list = batch_info_list + term_chunks_annotated

    batch_info_path = OUTPUT_DIR / "batch_request_info.json"
    with batch_info_path.open("w") as f:
        json.dump(batch_info_list, f)

    logger.info("make batch request")
    super_chunks = py_.chunk(batch_info_list, size=SUPER_CHUNK_SIZE)
    ic(len(batch_info_list))
    ic(len(super_chunks))
    for super_idx, chunks in enumerate(super_chunks):
        logger.info(f"super chunk #{super_idx}")
        make_batch_request(chunks=chunks)
    # term_super_chunks = py_.chunk(term_chunks_annotated, size=SUPER_CHUNK_SIZE)


if __name__ == "__main__":
    main()
