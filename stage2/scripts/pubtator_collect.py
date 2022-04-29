import time
from typing import List, Optional

import pandas as pd
import requests
from loguru import logger
from pydash import py_

from funcs import utils

ANNOTATE_URL_TEMPLATE = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/submit/{concept_type}"
SESSION_URL_TEMPLATE = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/retrieve/{session_id}"

data_root = utils.find_data_root()
assert data_root.exists()
OUTPUT_DIR = data_root / "output" / "pubtator"
BATCH_REQUEST_DIR = OUTPUT_DIR / "batch_requests"
BATCH_REQUEST_DIR.mkdir(exist_ok=True, parents=True)

ECHO_STEP = 10
TRY_INTERVAL = 5 * 60
MAX_NUM_TRIES = 5
CHUNK_SIZE = 4
CHUNK_SLEEP = 1


def sanitize_session_id(session_id: str) -> Optional[str]:
    session_id_length = 19  # e.g. "9359-6015-1307-8701"
    if len(session_id) != session_id_length:
        # ie api error
        return None
    else:
        return session_id


def retrieve_session(session_id: str) -> str:
    url = SESSION_URL_TEMPLATE.format(session_id=session_id)
    r = requests.get(url)
    r.raise_for_status()
    return r.text


def batch_retrieve_session(session_id_list: List[str]) -> None:
    output_files_exist = [
        (BATCH_REQUEST_DIR / f"{_}.txt").exists() for _ in session_id_list
    ]
    all_done = sum(output_files_exist) == len(output_files_exist)
    if all_done:
        return
    for nth_try in range(MAX_NUM_TRIES):
        for session_id in session_id_list:
            output_file = BATCH_REQUEST_DIR / f"{session_id}.txt"
            if output_file.exists():
                break
            time.sleep(CHUNK_SLEEP)
            session_results = None
            try:
                session_results = retrieve_session(session_id)
            except:
                continue
            if session_results is not None:
                with output_file.open("w") as f:
                    f.write(session_results)


def main():
    session_df_sanitized_path = OUTPUT_DIR / "session_df_sanitized.csv"
    if session_df_sanitized_path.exists():
        session_df = pd.read_csv(session_df_sanitized_path)
    else:
        session_df_path = OUTPUT_DIR / "session_df.csv"
        assert session_df_path.exists()
        session_df = pd.read_csv(session_df_path)
        session_df = (
            session_df.assign(
                session_id=lambda df: df["session_id"].apply(sanitize_session_id)
            )
            .dropna()
            .reset_index(drop=True)
        )
        session_df.to_csv(session_df_sanitized_path, index=False)

    session_list = session_df.to_dict(orient="records")
    session_chunks = py_.chunk(session_list, size=CHUNK_SIZE)
    for chunk_idx, chunk in enumerate(session_chunks):
        if chunk_idx % ECHO_STEP == 0:
            logger.info(f"#{chunk_idx} / {len(session_chunks)}")
        session_id_list = [_["session_id"] for _ in chunk]
        batch_retrieve_session(session_id_list=session_id_list)
        # time.sleep(CHUNK_SLEEP)


if __name__ == "__main__":
    main()
