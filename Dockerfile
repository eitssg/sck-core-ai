###############################################
# SCK Core AI Container (Local Dev / Fargate)
#
# Notes:
# - Keeps base image slim.
# - Uses uv for fast, deterministic installs.
# - Does NOT bundle optional extras (ai, vectordb) unless provided as build arg.
# - Designed for local iteration first; promote only after testing.
###############################################

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  UV_SYSTEM_PYTHON=1 \
  APP_HOME=/app

WORKDIR $APP_HOME

# Install system deps (add more only when needed)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  git curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install uv (fast dependency installer) + hatch (build backend)
RUN pip install --no-cache-dir uv hatchling

# When building from the monorepo root (as in docker compose using context ../..),
# the project files reside under a subdirectory (default: sck-core-ai). Allow override.
ARG PROJECT_SUBDIR=sck-core-ai

# Copy minimal metadata first for layer caching
COPY ${PROJECT_SUBDIR}/pyproject.toml pyproject.toml
COPY ${PROJECT_SUBDIR}/README.md README.md

# Build arg to include optional extras (e.g. ai, vectordb)
ARG EXTRAS=""

# Bring in local framework dependency first (needed for editable install reliability)
COPY ${PROJECT_SUBDIR}/../sck-core-framework ./sck-core-framework
RUN pip install ./sck-core-framework || (echo "Failed to install sck-core-framework" && exit 1)

# Install project (runtime only). Examples:
#   docker build --build-arg EXTRAS="[ai]" .
RUN if [ -n "$EXTRAS" ]; then \
  echo "Installing with extras $EXTRAS"; \
  uv pip install ".[${EXTRAS}]"; \
  else \
  uv pip install .; \
  fi

# Copy source AFTER deps to leverage cache (only project code; do NOT bundle external workbench flows)
COPY ${PROJECT_SUBDIR}/core_ai ./core_ai

# Non-root user
RUN useradd -m sck && chown -R sck:sck $APP_HOME
USER sck

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
  CMD curl -fsS http://localhost:8080/ready || exit 1

ENV SCK_AI_HOST=0.0.0.0 \
  SCK_AI_PORT=8080 \
  LANGFLOW_BASE_URL=http://host.docker.internal:7860

CMD ["python", "-m", "core_ai.server"]
