from enum import Enum

from environs import Env

env = Env()
env.read_env()

api_env = env.str("API_ENV", "dev")

resource_apis = {
    "the_bert": {
        "host": "http://localhost",
        "port": "8560",
        "ping": "http://localhost:8560/ping",
        "resource_url": "http://localhost:8570",
    },
    "vec": {
        "host": "http://localhost:9090",
        "port": "9090",
        "ping": "http://localhost:9090/status/",
        "resource_url": "http://localhost",
    },
}

bert_models = [
    "NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12",
    "biobert_v1.1_pubmed",
]

vec_models = ["BioSentVec"]

models = vec_models + bert_models


class ModelName(str, Enum):
    biosentvec = "BioSentVec"
    ncbi_bert_pubmed_mimic_uncased_base = (
        "NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12"
    )
    biobert_pubmed = "biobert_v1.1_pubmed"
