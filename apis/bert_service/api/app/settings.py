from environs import Env

env = Env()
env.read_env()

web_domain = env.str("WEB_DOMAIN", "localhost")
web_port = env.str("WEB_PORT", "8080")

bert_config_prod = [
    {
        "ip": "bert1",
        "port": 5555,
        "port_out": 5556,
        "label": "NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12",
    },
    {
        "ip": "bert2",
        "port": 5555,
        "port_out": 5556,
        "label": "biobert_v1.1_pubmed",
    },
]

bert_config_dev = [
    {"ip": "bert", "port": 5555, "port_out": 5556, "label": "bert_model"}
]

api_env = env.str("API_ENV", "dev")
if api_env == "prod":
    bert_config = bert_config_prod
    available_models = [item["label"] for item in bert_config_prod]
else:
    bert_config = bert_config_dev
    available_models = [item["label"] for item in bert_config_dev]
