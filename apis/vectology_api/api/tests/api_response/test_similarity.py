import json

import pytest
from loguru import logger
from starlette.testclient import TestClient

from app import settings
from app.main import app

client = TestClient(app)
url = "/cosine_similarity"


@pytest.mark.parametrize(
    "text_1, text_2, model_name",
    [
        ("Body mass index", "Coronary heart disease", model_name)
        for model_name in settings.models
    ],
)
def test_get_cosine_sim(text_1, text_2, model_name):
    payload = {"text_1": text_1, "text_2": text_2, "model_name": model_name}
    response = client.get(url, params=payload)
    res = response.json()
    assert response.status_code == 200
    assert list(res.keys()) == ["cosine_similarity"]
    assert type(res["cosine_similarity"]) == float


def test_get_cosine_sim_default():
    text_1 = "Body mass index"
    text_2 = "Coronary heart disease"
    payload0 = {"text_1": text_1, "text_2": text_2}
    response0 = client.get(url, params=payload0)
    res0 = response0.json()
    payload1 = {
        "text_1": text_1,
        "text_2": text_2,
        "model_name": settings.models[0],
    }
    response1 = client.get(url, params=payload1)
    res1 = response1.json()
    assert response0.status_code == 200
    assert res0["cosine_similarity"] == res1["cosine_similarity"]


@pytest.mark.parametrize(
    "model_name", [model_name for model_name in settings.models]
)
def test_post_cosine_sim(model_name):
    payload = {
        "text_list": ["Body mass index", "Coronary heart disease"],
        "model_name": model_name,
    }
    response = client.post(url, data=json.dumps(payload))
    res = response.json()
    logger.info(res)
    assert response.status_code == 200
    assert list(res.keys()) == ["pairs", "heatmap_data"]


def test_post_cosine_sim_exception_item_length():
    payload = {
        "text_list": [
            "item0",
            "item1",
            "item2",
            "item3",
            "item4",
            "item5",
            "item6",
            "item7",
            "item8",
            "item9",
            "item10",
        ],
        "model_name": "biobert_v1.1_pubmed",
    }
    response = client.post(url, data=json.dumps(payload))
    res = response.json()
    assert response.status_code == 400
    assert res == {"detail": "Too many items. Limit: 9."}
