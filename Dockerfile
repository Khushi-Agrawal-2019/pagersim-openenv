# FROM python:3.11-slim

# # Security: run as non-root user
# RUN useradd -m -u 1000 appuser

# WORKDIR /app

# # Install dependencies first (Docker layer caching)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy all source files
# COPY --chown=appuser:appuser . .

# # Switch to non-root user
# USER appuser

# # Expose HF Spaces port
# EXPOSE 7860

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
#     CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# CMD ["python3", "-m", "uvicorn", "api.server:app", \
#      "--host", "0.0.0.0", \
#      "--port", "7860", \
#      "--workers", "1", \
#      "--log-level", "info"]


FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 7860

# Start the unified FastAPI + Gradio app on port 7860
CMD ["python3", "app.py"]