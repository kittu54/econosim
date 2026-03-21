# EconoSim — Multi-agent economic simulation platform
# Two-stage build: install deps, then copy source

FROM python:3.11-slim AS base

LABEL maintainer="EconoSim Team"
LABEL description="Multi-agent economic simulation with Streamlit dashboard"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[viz]" 2>/dev/null || \
    pip install --no-cache-dir \
    numpy pandas pydantic pyyaml scipy \
    streamlit plotly matplotlib httpx

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Streamlit config
RUN mkdir -p /root/.streamlit
RUN echo '\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0"]
