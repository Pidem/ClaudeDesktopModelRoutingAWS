# Build/run the gateway with uv on a slim Python base.
FROM python:3.12-slim

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps first (cached layer) from lockfile.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# App code. inference_profiles.json is the (strategy x model) -> AIP ARN map the
# gateway loads; a {} placeholder is committed so builds before profiles exist
# still work (rebuild after running infra/create_inference_profiles.py).
COPY main.py config.py cost_ledger.py pricing.yaml ./
COPY router ./router
COPY infra/inference_profiles.json ./infra/inference_profiles.json

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080

# Health check hits /health; bind to 8080.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
