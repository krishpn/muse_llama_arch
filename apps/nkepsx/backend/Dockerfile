FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

ENV UV_COMPILE_BYTECODE=1

COPY apps/nkepsx/backend/pyproject.toml apps/nkepsx/backend/uv.lock /workspace/apps/nkepsx/backend/
COPY apps/open-aether/pyproject.toml* /workspace/apps/open-aether/
COPY apps/open-nrb/pyproject.toml* /workspace/apps/open-nrb/

WORKDIR /workspace/apps/nkepsx/backend

RUN uv sync --frozen --no-install-project --no-dev

COPY apps /workspace/apps

RUN uv pip install -e /workspace/apps/open-aether \
                   -e /workspace/apps/open-nrb \
                   -e /workspace/apps/nkepsx/backend

ENV PYTHONPATH="/workspace/apps/nkepsx/backend/src:/workspace/apps/open-aether/src:/workspace/apps/open-nrb/src"

EXPOSE 8000

CMD [".venv/bin/uvicorn", "nkepsx.main:app", "--host", "0.0.0.0", "--port", "8000"]
