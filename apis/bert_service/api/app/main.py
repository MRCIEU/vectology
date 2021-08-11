from fastapi import FastAPI
from loguru import logger
from starlette.middleware.cors import CORSMiddleware

from . import settings
from .apis import encode, similarity, status

logger.info(
    f"""Configs:
    - api_env: {settings.api_env}
    - web_domain: {settings.web_domain}
    - web_port: {settings.web_port}
"""
)

app = FastAPI(title="the BERT API", docs_url="/")

origins = [
    "http://localhost",
    # Current frontend port
    f"http://{settings.web_domain}:{settings.web_port}",
    # local dev
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(encode.router, tags=["model"])
app.include_router(similarity.router, tags=["model"])
app.include_router(status.router)
