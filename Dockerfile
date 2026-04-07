FROM python:3.11-slim

# Install uv first (as root)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/bin/uv

# Create user and setup app directory
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN chown appuser:appuser /app

# Switch to the non-root user for all further operations
USER appuser

# Configure uv for Docker compatibility
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies as appuser (creates .venv with correct permissions)
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy the rest of the source code
COPY --chown=appuser:appuser . .

# Final project install
RUN uv sync --frozen --no-dev

EXPOSE 7860

# Start the unified FastAPI + Gradio app via the 'server' script entry point
CMD ["uv", "run", "server"]