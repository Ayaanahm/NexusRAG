FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models \
    NEXUS_DATA_DIR=/data

WORKDIR /app
COPY pyproject.toml ./
COPY nexusrag ./nexusrag
COPY eval ./eval
RUN pip install --no-cache-dir . \
 && useradd --create-home --uid 1000 app \
 && mkdir -p /models /data && chown app:app /models /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"
CMD ["uvicorn", "nexusrag.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
