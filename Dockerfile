###############################################
# SCK Core AI Container (Simplified Editable Dev Image)
#
# Single-stage build using editable installs for both framework
# and AI project directly from source inside the container for AI). 
# Fast to understand, easy to tweak.
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

RUN pip install --no-cache-dir uv 

ARG EXTRAS="ai,vectordb"

# --- Framework (editable source) ---
COPY ./sck-core-framework ./sck-core-framework
COPY ./sck-core-db ./sck-core-db
COPY ./sck-core-execute ./sck-core-execute
COPY ./sck-core-runner ./sck-core-runner
COPY ./sck-core-deployspec ./sck-core-deployspec
COPY ./sck-core-component ./sck-core-component
COPY ./sck-core-invoker ./sck-core-invoker
COPY ./sck-core-organization ./sck-core-organization
COPY ./sck-core-api ./sck-core-api
COPY ./sck-core-codecommit ./sck-core-codecommit
COPY ./sck-core-cli ./sck-core-cli
COPY ./sck-core-ui ./sck-core-ui
COPY ./sck-core-ai ./sck-core-ai

WORKDIR ${APP_HOME}/sck-core-ai

# Install AI project (optionally extras) in editable mode.
RUN uv pip install -e .[${EXTRAS}]

# Remove the project folders now that the dependencies are installed
RUN rm -rf ${FRAMEWORK_SUBDIR} ${PROJECT_SUBDIR}

FROM base AS final

COPY ./sck-core-ai ./sck-core-ai

WORKDIR ${APP_HOME}/sck-core-ai

RUN uv sync --extras=${EXTRAS}

RUN useradd -m sck && chown -R sck:sck /app
USER sck

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s CMD curl -fsS http://localhost:8080/ready || exit 1

ENV SCK_AI_HOST=0.0.0.0 \
  SCK_AI_PORT=8080 \
  LANGFLOW_BASE_URL=http://host.docker.internal:7860

CMD ["python", "-m", "core_ai.server"]
