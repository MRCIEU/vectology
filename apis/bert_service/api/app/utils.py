from typing import Optional

from fastapi import HTTPException
from loguru import logger

import app.settings as settings


def select_bert(model_name: Optional[str]):
    logger.info(f"model_name: {model_name}")

    if model_name is None:
        res = (
            settings.bert_config[0]["ip"],
            settings.bert_config[0]["port"],
            settings.bert_config[0]["port_out"],
        )
    elif model_name not in settings.available_models:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    else:
        idx = settings.available_models.index(model_name)
        item = settings.bert_config[idx]
        res = (item["ip"], item["port"], item["port_out"])

    logger.info(f"bert_config: {res}")
    return res
