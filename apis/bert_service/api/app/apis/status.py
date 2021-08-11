import socket
from typing import List

from fastapi import APIRouter

from app import settings

router = APIRouter()


@router.get("/ping")
def get_ping() -> str:
    model_state = [
        check_model_connection(item["ip"], item["port"])
        for item in settings.bert_config
    ]
    model_connected = sum(model_state) == len(settings.bert_config)
    if model_connected:
        return f"Connected to the API and {sum(model_state)} models."
    else:
        return "Connected to the API, model unknown."


@router.get("/models")
def get_models() -> List[str]:
    """Get currently available models."""
    return settings.available_models


def check_model_connection(ip: str, port: int) -> bool:
    # https://gist.github.com/betrcode/0248f0894013382d7
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, port))
        s.shutdown(2)
        return True
    except:  # noqa
        return False
