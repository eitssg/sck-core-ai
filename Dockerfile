###############################################
# SCK Core AI Container (Simplified Editable Dev Image)
#
# Single-stage build using editable installs for both framework
# and AI project directly from source (no wheels, no Poetry usage
# inside the container for AI). Fast to understand, easy to tweak.
###############################################

FROM python:3.12-slim AS base 

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  UV_SYSTEM_PYTHON=1 \
  APP_HOME=/app

WORKDIR $APP_HOME

RUN apt-get update \
  && apt-get install -y --no-install-recommends git curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# Use uv for speed; no poetry needed for AI project itself.
RUN pip install --no-cache-dir uv poetry poetry-dynamic-versioning

ARG FRAMEWORK_SUBDIR=sck-core-framework
ARG PROJECT_SUBDIR=sck-core-ai
ARG EXTRAS="ai,vectordb"

# --- Framework (editable source) ---
COPY ${FRAMEWORK_SUBDIR} ./sck-core-framework
COPY ${PROJECT_SUBDIR} ./sck-core-ai

WORKDIR ${FRAMEWORK_SUBDIR}

ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_VIRTUALENVS_IN_PROJECT=false
ENV POETRY_DYNAMIC_VERSIONING_ENABLE=0
ENV POETRY_NO_INTERACTION=1 
ENV POETRY_DYNAMIC_VERSIONING_BYPASS=1
RUN poetry install --no-root

WORKDIR ${APP_HOME}/${PROJECT_SUBDIR}

# Install AI project (optionally extras) in editable mode.
RUN if [ -n "$EXTRAS" ]; then \
  echo "Installing AI project with extras $EXTRAS"; \
  uv pip install -e .[${EXTRAS}] ; \
  else \
  uv pip install -e . ; \
  fi

# Remove the project folders now that the dependencies are installed
RUN rm -rf ${FRAMEWORK_SUBDIR} ${PROJECT_SUBDIR}

FROM base AS final

COPY ${FRAMEWORK_SUBDIR} ./sck-core-framework
COPY ${PROJECT_SUBDIR} ./sck-core-ai

WORKDIR ${FRAMEWORK_SUBDIR}
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=false
ENV POETRY_DYNAMIC_VERSIONING_ENABLE=0
RUN poetry install

WORKDIR ${APP_HOME}/${PROJECT_SUBDIR}
RUN uv pip install -e .[${EXTRAS}]

RUN useradd -m sck && chown -R sck:sck /app
USER sck

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s CMD curl -fsS http://localhost:8080/ready || exit 1

ENV SCK_AI_HOST=0.0.0.0 \
  SCK_AI_PORT=8080 \
  LANGFLOW_BASE_URL=http://host.docker.internal:7860

CMD ["python", "-m", "core_ai.server"]
