# Main API

## Usage

```
# A user might need to tweak various configs, e.g. ports
docker-compose up -d
```

## File structure

A complete project is expected to look like the following

```
.
├── api
│   ├── app
│   │   ├── apis
│   │   │   ├── ati.py
│   │   │   ├── encode.py
│   │   │   ├── gwas_db.py
│   │   │   ├── __init__.py
│   │   │   ├── preprocess.py
│   │   │   ├── similarity.py
│   │   │   └── status.py
│   │   ├── funcs
│   │   │   ├── __init__.py
│   │   │   ├── preprocess.py
│   │   │   ├── ukbb_filtering_rules.py
│   │   │   └── utils.py
│   │   ├── gunicorn_conf.py
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── settings.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── logging.py
│   │       └── models.py
│   ├── data
│   ├── Dockerfile
│   ├── __init__.py
│   ├── Makefile
│   ├── mypy.ini
│   ├── poetry.lock
│   ├── pyproject.toml
│   ├── tests
│   │   ├── api_response
│   │   │   ├── __init__.py
│   │   │   ├── test_encode.py
│   │   │   ├── test_preprocess.py
│   │   │   ├── test_similarity.py
│   │   │   └── test_status.py
│   │   └── funcs
│   │       ├── __init__.py
│   │       ├── test_preprocess.py
│   │       └── test_utils.py
│   └── tox.ini
├── docker-compose.yml
└── README.md
```
