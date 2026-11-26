FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1
COPY requirements.txt packages.txt .
RUN apt-get update && xargs apt-get install -y --no-install-recommends < packages.txt \
    && rm -rf /var/lib/apt/lists/* && pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["sh", "-c", "exec python -m streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.headless true"]
