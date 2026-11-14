FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV CHROMA_PATH=demo_chroma_db
ENV CHUNKS_DIR=demo_data/chunks
ENV CHROMA_RUNTIME_COPY=true
ENV ENABLE_LIVE_INGEST=false
ENV MANDI_PRIMARY_PROVIDER=agmarknet_public

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar tesseract-ocr-pan \
    fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python build_model.py

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
