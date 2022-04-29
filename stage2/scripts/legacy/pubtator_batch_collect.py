import json

import requests

from funcs import utils
from funcs.utils import ic

SESSION_URL_TEMPLATE = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/retrieve/{session_id}"

data_root = utils.find_data_root()
assert data_root.exists()
OUTPUT_DIR = data_root / "output" / "pubtator"
BATCH_REQUEST_DIR = OUTPUT_DIR / "batch_requests"
BATCH_REQUEST_DIR.mkdir(exist_ok=True, parents=True)


def main():
    session_info_path = OUTPUT_DIR / "batch_request_info.json"
    assert session_info_path.exists()
    with session_info_path.open() as f:
        batch_info_list = json.load(f)

    for info in batch_info_list:
        ic("{batch} {idx}".format(batch=info["batch"], idx=info["idx"]))
        session_id = info["session_id"]
        url = SESSION_URL_TEMPLATE.format(session_id=session_id)
        r = requests.get(url)
        try:
            r.raise_for_status()
            ic("{batch} {idx} {session_id}".format(batch=info["batch"], idx=info["idx"], session_id=session_id))
        except Exception as e:
            print(e)
            continue
        results = r.text
        output_path = BATCH_REQUEST_DIR / f"{session_id}"
        with output_path.open("w") as f:
            f.write(results)


if __name__ == "__main__":
    main()
