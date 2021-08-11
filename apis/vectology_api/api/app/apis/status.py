from typing import Dict, List

import requests
from fastapi import APIRouter
from pydantic import BaseModel

from app import settings

router = APIRouter()


class ResourceStatus(BaseModel):
    url: str
    status: bool


@router.get("/ping", response_model=Dict[str, ResourceStatus])
def get_ping():
    """Check the status of associated resources.
    """
    resource_state = {
        item_key: {
            "url": item["resource_url"],
            "status": ping_status(item["ping"]),
        }
        for (item_key, item) in settings.resource_apis.items()
    }
    return resource_state


@router.get("/models", response_model=List[str])
def get_models():
    """Returns the names of the associated models
    """
    return settings.models


def ping_status(url: str) -> bool:
    response = requests.get(url)
    res = response.status_code == 200
    return res
