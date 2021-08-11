import json
from typing import List, Optional

import requests
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, validator

from app import settings

router = APIRouter()


class EncodeResponse(BaseModel):
    embeddings: List[List[float]]


class EncodeInput(BaseModel):
    text_list: List[str]
    model_name: settings.ModelName

    @validator("text_list", whole=True)
    def text_list_length(cls, v):
        limit = 50
        if len(v) > limit:
            raise HTTPException(
                status_code=400, detail=f"Too many items. Limit: {limit}."
            )
        return v


@router.get("/encode", response_model=EncodeResponse)
def get_encode(text: str, model_name: Optional[settings.ModelName] = None):
    """Compute the embeddings for `text`
    """
    logger.info(f"text: {text}\tmodel_name: {model_name}")
    if model_name is None:
        model_name = settings.ModelName.ncbi_bert_pubmed_mimic_uncased_base

    if model_name.value in settings.bert_models:
        res = get_encode_bert(text=text, model_name=model_name.value)
    elif model_name.value in settings.vec_models:
        text_list = [text]
        res = post_encode_vec(text_list=text_list, model_name=model_name.value)
    return res


@router.post("/encode", response_model=EncodeResponse)
def post_encode(input: EncodeInput):
    """Compute the embeddings for `text_list`

    Limits:
    - `text_list`: Number of items should not exceed 50.
    """
    logger.info(f"input: {input}")
    text_list = input.text_list
    model_name = input.model_name

    if model_name.value in settings.bert_models:
        res = post_encode_bert(text_list=text_list, model_name=model_name)
    elif model_name.value in settings.vec_models:
        res = post_encode_vec(text_list=text_list, model_name=model_name)
    return res


def get_encode_bert(text: str, model_name: str):
    resource = settings.resource_apis["the_bert"]
    url = f"{resource['host']}:{resource['port']}/encode"
    payload = {"text": text, "model_name": model_name}
    response = requests.get(url, params=payload)
    res = response.json()
    return res


def post_encode_bert(text_list: List[str], model_name: str):
    resource = settings.resource_apis["the_bert"]
    url = f"{resource['host']}:{resource['port']}/encode"
    payload = {"text_list": text_list, "model_name": model_name}
    response = requests.post(url, data=json.dumps(payload))
    res = response.json()
    return res


def post_encode_vec(text_list: List[str], model_name: str):
    resource = settings.resource_apis["vec"]
    url = f"{resource['host']}/sent2vec-encode/"
    payload = {"text": text_list}
    response = requests.post(url, data=json.dumps(payload))
    res = response.json()
    return res
