# InvoAgent: the web app with the Tesseract OCR program, for Render or any Docker host.
#   docker build -t invoagent .
#   docker run -p 5077:5077 -e PORT=5077 -e INVOICE_ADMIN_USER=admin -e INVOICE_ADMIN_PASSWORD=... -v invoagent-data:/var/data invoagent
FROM python:3.12-slim-bookworm

# Tesseract OCR, with the same English model as the tested Windows install (tessdata_fast, checked by hash)
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*
ADD https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata /usr/share/tessdata-fast/eng.traineddata
RUN echo "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2  /usr/share/tessdata-fast/eng.traineddata" \
    | sha256sum -c - && chmod 644 /usr/share/tessdata-fast/eng.traineddata
ENV TESSDATA_PREFIX=/usr/share/tessdata-fast

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY . .

# Data (database, stored invoices, users) lives in /var/data: mount a persistent disk there.
# One worker thread fits small cloud instances; raise INVOICE_WORKERS on bigger ones.
ENV PYTHONUNBUFFERED=1 \
    TESSERACT_CMD=/usr/bin/tesseract \
    INVOICE_DATA_DIR=/var/data \
    INVOICE_WORKERS=1 \
    USE_EMBEDDINGS=0 \
    OMP_THREAD_LIMIT=1 \
    PORT=10000

EXPOSE 10000
HEALTHCHECK --interval=60s --timeout=10s CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '10000'))"
CMD ["python", "app.py"]
