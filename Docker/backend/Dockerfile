# Build context is the repo root (see docker-compose.yml), so that Multiagent/ is importable as a package.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    HOME=/home/app

ENV LANGSMITH_TRACING=false

WORKDIR /app

RUN useradd --create-home --uid 10001 app

COPY Docker/backend/requirements.server.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY --chown=app:app Multiagent/ /app/Multiagent/

USER app

EXPOSE 8077

HEALTHCHECK --interval=15s --timeout=10s --start-period=90s --retries=10 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8077/v1/models', timeout=8).status == 200 else 1)"]

# Exactly one worker and no --reload: _quiz_cache is process-local, otherwise /quiz/submit ends in 404.
CMD ["uvicorn", "Multiagent.server.app:app", \
     "--host", "0.0.0.0", "--port", "8077", \
     "--workers", "1", "--proxy-headers"]
