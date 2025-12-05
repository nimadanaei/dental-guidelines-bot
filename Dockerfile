# Use official Python image
FROM python:3.11-slim

# Work inside /app
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Railway will inject PORT, but default to 8000 if missing
ENV PORT=8000

# Start FastAPI with Uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
