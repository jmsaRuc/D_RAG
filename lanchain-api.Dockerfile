FROM --platform=$BUILDPLATFORM python:3.13-slim-bookworm AS prod


ARG APP_HOME=/app


RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
curl \
wget \
gnupg \
git \
cmake \
pkg-config \
python3-dev \
libjpeg-dev \
&& apt-get clean \ 
&& rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get dist-upgrade -y \
&& rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

RUN mkdir -p /home/appuser && chown -R appuser:appuser /home/appuser 


# --- Install Poetry ---
ARG POETRY_VERSION=2.1.1
ENV POETRY_VIRTUALENVS_IN_PROJECT=false
ENV POETRY_VIRTUALENVS_CREATE=true
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_VIRTUALENVS_OPTIONS_SYSTEM_SITE_PACKAGES=ture
ENV POETRY_CACHE_DIR=/home/appuser/.cache/pypoetry
ENV POETRY_HOME=/home/appuser/.local/share/pypoetry
ENV POETRY_VIRTUALENVS_PATH=/home/appuser/.cache/pypoetry/virtualenvs
ENV PYTHONUNBUFFERED=1

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR ${APP_HOME}

# 1) Install dependencies
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-root

# 2) Copy the repository content
COPY . ${APP_HOME}/

# 3) Install the package
RUN poetry install

# 4) Install crawl4ai and playwright
RUN poetry run crawl4ai-setup
RUN poetry run playwright install --with-deps chromium

# 5) Copy the playwright cache to the non-root user
RUN mkdir -p /home/appuser/.cache/ms-playwright \
    && cp -r /root/.cache/ms-playwright/chromium-* /home/appuser/.cache/ms-playwright/ \
    && chown -R appuser:appuser /home/appuser/.cache/ms-playwright

# 6) Set permissions for the app directory
RUN chown -R appuser:appuser ${APP_HOME} \
    && chown -R appuser:appuser /home/appuser/

# 7) Set the user to the non-root user
USER appuser

# 8) Test crawl4ai
RUN poetry run crawl4ai-doctor

EXPOSE 2024

# 9) run the app

CMD ["poetry", "run",\
     "langgraph", \
     "dev", \
     "--allow-blocking",\
     "--host", "0.0.0.0"]