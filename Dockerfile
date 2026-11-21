FROM python:3.11-slim AS base
# xgboost needs libgomp (OpenMP), which the slim image doesn't ship
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Not `pip install .`: src/config.py finds artefacts/ relative to itself, so src must stay in /app

FROM base AS train
COPY src/ src/
COPY data/ data/
COPY config/ config/
COPY train.py .
RUN python train.py --skip-tuning

FROM base AS runtime
RUN useradd --create-home --uid 1000 app
COPY src/ src/
COPY dashboard/ dashboard/
COPY --from=train /app/artefacts/ artefacts/
# Fails the build on a bad pickle or missing libgomp, which the health check can't see
RUN python -m src.smoke_test
USER app
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
# The slim image has no curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
