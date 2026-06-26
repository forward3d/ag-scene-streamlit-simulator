# Build stage - includes build tools and dependencies
# Testing
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Set the working directory for the application.
WORKDIR /app

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Runtime stage - minimal image with only runtime dependencies
# UPDATE: Removed the hardcoded @sha256 to ensure it pulls the latest patched version to pass the security scan
# FROM registry.management.stage.lv-426.tech/dockerhub_cache/python:3.12-slim-bookworm
FROM python:3.12-slim-bookworm

ARG GIT_REPOSITORY_URL
ARG GIT_COMMIT_SHA
ENV DD_TAGS="git.repository_url:${GIT_REPOSITORY_URL},git.commit.sha:${GIT_COMMIT_SHA}"
ENV DD_GIT_REPOSITORY_URL=${GIT_REPOSITORY_URL}
ENV DD_GIT_COMMIT_SHA=${GIT_COMMIT_SHA}

# renovate: datasource=deb depName=g++ registryUrl=https://deb.debian.org/debian?release=bookworm&components=main&binaryArch=amd64
ENV G_PLUS_PLUS_VERSION="4:12.2.0-3"

RUN apt-get update -qq && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    g++=${G_PLUS_PLUS_VERSION} \
    && rm -rf /var/lib/apt/lists /var/cache/apt/archives

# Set the working directory for the application.
WORKDIR /app

# Run and own only the runtime files as a non-root user for security
RUN useradd appuser --create-home --shell /bin/bash && \
    chown -R appuser:appuser /app
USER appuser

# Copy the application from the builder
COPY --from=builder --chown=appuser:appuser /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Set the default command that runs when the container starts.
EXPOSE 8501
# UPDATE: Pointed to the new app.py file
CMD ["streamlit", "run", "code/app.py", "--server.port=8501", "--server.address=0.0.0.0"]