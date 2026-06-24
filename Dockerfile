FROM python:3.11-slim

WORKDIR /app

# Install deps first so Docker layer-caches them when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (including prebuilt embeddings.npy + metadata.pkl)
COPY . .

# Railway/most PaaS inject $PORT; default to 8000 locally. Shell form expands it.
ENV PORT=8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
