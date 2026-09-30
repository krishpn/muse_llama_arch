# Use official Python 3.12 slim image
FROM python:3.12-slim

# Install uv for ultra-fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock /app/

# Install dependencies using uv
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source code
COPY . /app

# Tell Python where to find the source packages
ENV PYTHONPATH=/app/src

# Run uvicorn directly from the pre-built virtual environment binary
EXPOSE 8000
CMD [".venv/bin/uvicorn", "nkepsx.main:app", "--host", "0.0.0.0", "--port", "8000"]