FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl cron tzdata && rm -rf /var/lib/apt/lists/*

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cache dirs (mounted as volumes)
RUN mkdir -p data/cache/nhl data/cache/mlb data/cache/nba data/cache/model logs

# Streamlit config
RUN mkdir -p /root/.streamlit
COPY docker/streamlit_config.toml /root/.streamlit/config.toml

# Cron
COPY docker/crontab /etc/cron.d/warm_cache
RUN chmod 0644 /etc/cron.d/warm_cache && crontab /etc/cron.d/warm_cache

COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["/start.sh"]
