FROM --platform=$BUILDPLATFORM python:3.12-slim-bullseye AS prod

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager (use pip for safer cross-arch install)
#RUN pip install --no-cache-dir --upgrade pip
#RUN pip install uv
#ENV PATH="/root/.local/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip
RUN pip install poetry==2.1.1

# Configuring poetry
RUN poetry config virtualenvs.create false
RUN poetry config cache-dir /tmp/poetry_cache

# Copying requirements of a project
COPY pyproject.toml poetry.lock /app/

WORKDIR /app

# Installing requirements
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --no-root --only main

# 2) Copy the repository content
COPY . /app
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main

# 3) Provide default environment variables to point to Ollama (running elsewhere)
#    Adjust the OLLAMA_URL to match your actual Ollama container or service.

# 4) Expose the port that LangGraph dev server uses (default: 2024)

# 5) Launch the assistant with the LangGraph dev server:
#    Equivalent to the quickstart: uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
CMD ["uvx", \
     "--refresh", \
     "--from", "langgraph-cli[inmem]", \
     "--with-editable", ".", \
     "--python", "3.12", \
     "langgraph", \
     "dev", \
     "--host", "0.0.0.0"]

FROM prod AS dev

RUN --mount=type=cache,target=/tmp/poetry_cache poetry install