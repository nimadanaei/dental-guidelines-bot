# Use official Python image
FROM python:3.11-slim

# Work inside /app
WORKDIR /app

# Install Python dependencies directly (no requirements.txt needed)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    numpy \
    pydantic \
    pypdf \
    "openai>=1.0.0"

# Copy all project files into the image
COPY . .

# Expose the port Railway will hit
ENV PORT=8000

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]