from loguru import logger
from starlette.testclient import TestClient

from app import settings
from app.main import app

client = TestClient(app)


def test_get_ping():
    url = "/ping"
    response = client.get(url)
    res = response.json()
    logger.info(res)
    expected_res = {
        item_key: {"url": item["resource_url"], "status": True}
        for (item_key, item) in settings.resource_apis.items()
    }
    assert res == expected_res


def test_get_models():
    url = "/models"
    response = client.get(url)
    res = response.json()
    logger.info(res)
    expected_res = settings.models
    assert res == expected_res
