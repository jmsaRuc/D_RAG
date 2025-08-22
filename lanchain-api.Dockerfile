FROM --platform=$BUILDPLATFORM python:3.13-slim-bookworm AS builder

ARG APP_HOME=/app

ENV POETRY_VERSION=2.1.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR ${APP_HOME}

# Copying requirements of a project
COPY pyproject.toml poetry.lock ${APP_HOME}/
RUN touch README.md
RUN touch LICENSE

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root --only main

RUN rm -rf $POETRY_CACHE_DIR


FROM --platform=$BUILDPLATFORM python:3.13-slim-bookworm AS runtime

ARG APP_HOME=/app
ENV POETRY_VERSION=2.1.1

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Copying the rest of the project files

COPY pyproject.toml poetry.lock ${APP_HOME}/
COPY --from=builder ${APP_HOME}/.venv ${VIRTUAL_ENV}


COPY langgraph.json ${APP_HOME}/

COPY src ${APP_HOME}/src/

WORKDIR ${APP_HOME}



RUN touch README.md
RUN touch LICENSE

RUN pip install "poetry==${POETRY_VERSION}" \
    && poetry install --only main \
    && pip uninstall poetry -y \
    && rm -rf $POETRY_CACHE_DIR
    
RUN crawl4ai-setup
RUN playwright install --with-deps chromium
RUN crawl4ai-doctor

EXPOSE 2024

CMD ["langgraph", \
     "dev", \
     "--allow-blocking",\
     "--host", "0.0.0.0"]