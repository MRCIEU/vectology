FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./Pipfile .
COPY ./Pipfile.lock .
RUN pip install pipenv
RUN pipenv install --system

COPY ./ /app
