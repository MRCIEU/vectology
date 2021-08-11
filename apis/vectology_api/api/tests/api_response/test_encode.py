import json

import pytest
from loguru import logger
from starlette.testclient import TestClient

from app import settings
from app.main import app

client = TestClient(app)
url = "/encode"


@pytest.mark.parametrize(
    "text, model_name",
    [("Body mass index", model_name) for model_name in settings.models],
)
def test_get_encode(text, model_name):
    payload = {"text": text, "model_name": model_name}
    response = client.get(url, params=payload)
    res = response.json()
    logger.info(res)
    assert response.status_code == 200
    assert list(res.keys()) == ["embeddings"]
    assert len(res["embeddings"]) >= 1


def test_get_encode_default():
    text = "Body mass index"
    payload0 = {"text": text}
    response0 = client.get(url, params=payload0)
    res0 = response0.json()
    payload1 = {"text": text, "model_name": settings.models[0]}
    response1 = client.get(url, params=payload1)
    res1 = response1.json()
    assert response0.status_code == 200
    assert res0["embeddings"] == res1["embeddings"]


@pytest.mark.parametrize(
    "text_list, model_name",
    [
        (["Body mass index", "Coronary heart disease"], model_name)
        for model_name in settings.models
    ],
)
def test_post_encode(text_list, model_name):
    payload = {"text_list": text_list, "model_name": model_name}
    response = client.post(url, data=json.dumps(payload))
    res = response.json()
    logger.info(res)
    assert response.status_code == 200
    assert list(res.keys()) == ["embeddings"]
    assert len(res["embeddings"]) >= 1
