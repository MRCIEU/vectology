from typing import List

from pydantic import BaseModel


class BertEncodeResponse(BaseModel):
    embeddings: List[List[float]]


class SingleCosineSim(BaseModel):
    cosine_similarity: float


class MultiCosineSim(BaseModel):
    idx1: int
    idx2: int
    cosine_similarity: float


class TextList(BaseModel):
    text_list: List[str]
    model_name: str
