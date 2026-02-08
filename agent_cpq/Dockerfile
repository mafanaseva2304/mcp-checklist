FROM python:3.12-slim

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false --local && \
    poetry install --no-root --no-interaction

COPY . /app

EXPOSE 8001

CMD ["python", "main.py"]