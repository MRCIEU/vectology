from itertools import combinations
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, validator

from app import settings
from app.apis.encode import post_encode_bert, post_encode_vec
from app.funcs.utils import cosine_sim, nest_pairs, pairwise_cosine_sim

router = APIRouter()


class SimilarityInput(BaseModel):
    text_list: List[str]
    model_name: settings.ModelName

    @validator("text_list", whole=True)
    def text_list_length(cls, v):
        limit = 9
        if len(v) > limit:
            raise HTTPException(
                status_code=400, detail=f"Too many items. Limit: {limit}."
            )
        return v


class SingleCosineSim(BaseModel):
    cosine_similarity: Optional[float]


class CosineSim(BaseModel):
    idx1: int
    idx2: int
    cosine_similarity: Optional[float]


class MultiCosineSim(BaseModel):
    pairs: List[CosineSim]
    heatmap_data: Any


@router.get("/cosine_similarity", response_model=SingleCosineSim)
def get_cosine_sim(
    text_1: str, text_2: str, model_name: Optional[settings.ModelName] = None
):
    """Compute cosine similarity (1 - cosine distance)
    between the embeddings of `text_1` and `text_2`
    """
    logger.info(
        f"""
    - text_1: {text_1}
    - text_2: {text_2}
    - model_name: {model_name}
    """
    )
    if model_name is None:
        model_name = settings.ModelName.biosentvec
    text_list = [text_1, text_2]

    if model_name.value in settings.bert_models:
        encode_res = post_encode_bert(
            text_list=text_list, model_name=model_name.value
        )
        embeddings = encode_res["embeddings"]
    if model_name.value in settings.vec_models:
        encode_res = post_encode_vec(
            text_list=text_list, model_name=model_name.value
        )
        embeddings = encode_res["embeddings"]
    res = {"cosine_similarity": cosine_sim(embeddings[0], embeddings[1])}
    return res


@router.post("/cosine_similarity", response_model=MultiCosineSim)
def post_cosine_sim(input: SimilarityInput):
    """Compute pairwise cosine similarity between embeddings for
    elements in `text_list`.

    For `text_list` of length $N$, the pairs has length
    $N! / ((N - 2)! * 2!)$.

    Limits:
    - `text_list`: Number of items should not exceed 9.
    """
    logger.info(f"input: {input}")
    text_list = input.text_list
    model_name = input.model_name
    pairs = list(combinations(range(len(text_list)), 2))

    if model_name.value in settings.bert_models:
        encode_res = post_encode_bert(
            text_list=text_list, model_name=model_name.value
        )
        embeddings = encode_res["embeddings"]
    if model_name.value in settings.vec_models:
        encode_res = post_encode_vec(
            text_list=text_list, model_name=model_name.value
        )
        embeddings = encode_res["embeddings"]
    pairs = [
        pairwise_cosine_sim(pair=pair, embeddings=embeddings) for pair in pairs
    ]
    heatmap_data = nest_pairs(pairs)
    res = {"pairs": pairs, "heatmap_data": heatmap_data}
    return res
