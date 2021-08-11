# API to serve BERT models

## Usage

Obtain the following finetuned BERT models from the following repos 
and move them to relevant directories (check file structure):

- [BlueBERT](https://github.com/ncbi-nlp/bluebert)
- [BioBERT](https://github.com/dmis-lab/biobert)

Then run (a user might need to further tweak docker setup configs):

```
docker-compose up -d
```

## File structure

A complete project is expected to look like the following

```
 .
├── api
│   ├── app
│   │   ├── apis
│   │   │   ├── encode.py
│   │   │   ├── similarity.py
│   │   │   └── status.py
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models
│   │   │   └── models.py
│   │   ├── settings.py
│   │   └── utils.py
│   ├── Dockerfile
│   ├── __init__.py
│   ├── Makefile
│   ├── Pipfile
│   ├── Pipfile.lock
│   ├── pyproject.toml
│   └── tox.ini
├── bert
│   ├── docker
│   │   ├── Dockerfile-bert-cpu
│   │   ├── Dockerfile-bert-gpu
│   │   ├── entrypoint-bert-cpu.sh
│   │   └── entrypoint-bert-gpu.sh
│   └── models
│       ├── biobert_v1.1_pubmed
│       │   ├── bert_config.json
│       │   ├── bert_model.ckpt.index
│       │   ├── bert_model.ckpt.meta
│       │   └── vocab.txt
│       └── NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12
│           ├── bert_config.json
│           ├── bert_model.ckpt.index
│           ├── bert_model.ckpt.meta
│           └── vocab.txt
├── docker-compose.yml
└── README.md
```
