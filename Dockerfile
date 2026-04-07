FROM python:3.11-slim

RUN useradd -m -u 1000 appuser
WORKDIR /app

# Use uv for fast dependency management and compliance
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/bin/uv

# Prevent uv from using hardlinks, which can fail in some Docker environments
ENV UV_LINK_MODE=copy

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source and install project
COPY --chown=appuser:appuser . .
RUN uv sync --frozen --no-dev

USER appuser
EXPOSE 7860

# Start the unified FastAPI + Gradio app via the 'server' script entry point
CMD ["uv", "run", "server"]