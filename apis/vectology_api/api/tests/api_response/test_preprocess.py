# flake8: noqa
import json

from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)
url = "/preprocess"

orig_text_list = [
    # no associated pattern, should do nothing
    "Major Depressive Disorder",
    # rules that return "asis"
    "Current employment status: Doing unpaid or voluntary work",
    "Reason for glasses/contact lenses: For 'astigmatism'",
    # should return last matching group
    "Diagnoses - main ICD10: A02.0 Salmonella gastro-enteritis",
    "Diagnoses - main ICD10: A04.4 Other intestinal Escherichia coli infections",
    "Destinations on discharge from hospital (recoded): Usual Place of residence",
    "Medication for pain relief, constipation, heartburn: Laxatives (e.g. Dulcolax, Senokot)",
    # prepend last matching group with cancer
    "Cancer code, self-reported: acute myeloid leukaemia",
    "Type of cancer: ICD10: C00.0 External upper lip",
]

expected_filter_res = [
    # no associated pattern, should do nothing
    {"result": "major depressive disorder", "rule": "asis", "pattern": None},
    # rules that return "asis"
    {
        "result": "current employment status: doing unpaid or voluntary work",
        "rule": "asis",
        "pattern": "^current employment status: (.*)",
    },
    {
        "result": "reason for glasses/contact lenses: for 'astigmatism'",
        "rule": "asis",
        "pattern": "^reason for glasses/contact lenses: (.*)",
    },
    # should return last matching group
    {
        "result": "salmonella gastro-enteritis",
        "rule": "last_matching_group",
        "pattern": r"^diagnoses - (main|secondary) icd(9|10): \w+\.?\w* (.*)",
    },
    {
        "result": "other intestinal escherichia coli infections",
        "rule": "last_matching_group",
        "pattern": r"^diagnoses - (main|secondary) icd(9|10): \w+\.?\w* (.*)",
    },
    {
        "result": "usual place of residence",
        "rule": "last_matching_group",
        "pattern": r"^destinations on discharge from hospital \(recoded\): (.*)",
    },
    {
        "result": "laxatives (e.g. dulcolax, senokot)",
        "rule": "last_matching_group",
        "pattern": r"^medication for pain relief,? constipation,? heartburn: (.*)",
    },
    # prepend last matching group with cancer
    {
        "result": "cancer acute myeloid leukaemia",
        "rule": "last_matching_group_with_prefix",
        "pattern": r"^cancer code,? self-reported: (.*)",
    },
    {
        "result": "cancer external upper lip",
        "rule": "last_matching_group_with_prefix",
        "pattern": r"^type of cancer: icd10: \w+\.?\w* (.*)",
    },
]


def test_get_preprocess_ukbb():
    res = [
        client.get(url, params={"text": text}).json()
        for text in orig_text_list
    ]
    assert res == expected_filter_res


def test_post_preprocess_ukbb():
    response = client.post(
        url,
        data=json.dumps(
            {"text_list": orig_text_list, "source": "ukbb", "method": "auto"}
        ),
    )
    res = response.json()
    assert response.status_code == 200
    assert res == expected_filter_res


def test_post_preprocess_limit():
    text_list = ["foobar" for _ in range(51)]
    response = client.post(
        url,
        data=json.dumps(
            {"text_list": text_list, "source": "ukbb", "method": "auto"}
        ),
    )
    res = response.json()
    assert response.status_code == 400
    assert res == {"detail": "Too many items. Limit: 50."}
