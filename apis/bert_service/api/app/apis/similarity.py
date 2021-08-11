from itertools import combinations
from typing import List, Optional

import numpy as np
from bert_serving.client import BertClient
from fastapi import APIRouter, HTTPException
from scipy.spatial.distance import cosine

from app.models.models import MultiCosineSim, SingleCosineSim, TextList
from app.utils import select_bert

router = APIRouter()


@router.get("/cosine_similarity", response_model=SingleCosineSim)
def get_cosine_sim(text_1: str, text_2: str, model_name: Optional[str] = None):
    """Compute cosine similarity (1 - cosine distance)
    between the embeddings of `text_1` and `text_2`
    """
    bert_ip, bert_port, bert_port_out = select_bert(model_name)
    with BertClient(
        ip=bert_ip, port=bert_port, port_out=bert_port_out, output_fmt="list"
    ) as bc:
        embeddings = bc.encode([text_1, text_2])
        res = {"cosine_similarity": cosine_sim(embeddings[0], embeddings[1])}
    return res


@router.post("/cosine_similarity", response_model=List[MultiCosineSim])
def post_cosine_sim(input: TextList):
    """Compute pairwise cosine similarity between embeddings for
    elements in `text_list`.

    For `text_list` of length $N$, the pairs has length
    $N! / ((N - 2)! * 2!)$.

    NOTE: $N$ should be less than 10.
    """
    bert_ip, bert_port, bert_port_out = select_bert(input.model_name)
    if len(input.text_list) >= 10:
        raise HTTPException(
            status_code=400,
            detail="Number of elements in text_list should be less than 10.",
        )
    pairs = list(combinations(range(len(input.text_list)), 2))
    with BertClient(
        ip=bert_ip, port=bert_port, port_out=bert_port_out, output_fmt="list"
    ) as bc:
        embeddings = bc.encode(input.text_list)
        res = [
            pairwise_cosine_sim(pair=pair, embeddings=embeddings)
            for pair in pairs
        ]
    return res


def cosine_sim(x: List[float], y: List[float]):
    return 1 - cosine(np.array(x), np.array(y))


def pairwise_cosine_sim(pair, embeddings):
    idx1 = pair[0]
    idx2 = pair[1]
    res = {
        "idx1": idx1,
        "idx2": idx2,
        "cosine_similarity": cosine_sim(embeddings[idx1], embeddings[idx2]),
    }
    return res
