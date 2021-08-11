from typing import Optional

from bert_serving.client import BertClient
from fastapi import APIRouter
from loguru import logger

from app.models.models import BertEncodeResponse, TextList
from app.utils import select_bert

router = APIRouter()


@router.get("/encode", response_model=BertEncodeResponse)
def get_encode(text: str, model_name: Optional[str] = None):
    """Compute the embeddings for `text` as a $1 \times P$ vector,
    where $P$ is currently set to 768.
    """
    bert_ip, bert_port, bert_port_out = select_bert(model_name)
    with BertClient(
        ip=bert_ip, port=bert_port, port_out=bert_port_out, output_fmt="list"
    ) as bc:
        embeddings = bc.encode([text])
        res = {"embeddings": embeddings}
    return res


@router.post("/encode", response_model=BertEncodeResponse)
def post_encode(input: TextList):
    """Compute the embeddings for `text_list` as a $N \times P$ vector,
    where $N$ is the length of the list, and $P$ is currently set to 768.
    """
    logger.info(f"input: {input}")
    bert_ip, bert_port, bert_port_out = select_bert(input.model_name)
    with BertClient(
        ip=bert_ip, port=bert_port, port_out=bert_port_out, output_fmt="list"
    ) as bc:
        embeddings = bc.encode(input.text_list)
        res = {"embeddings": embeddings}
    return res
