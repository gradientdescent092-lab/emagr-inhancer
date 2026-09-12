FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Linux libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start Flask API
CMD ["sh", "-c", "gunicorn --timeout 600 --workers 1 --bind 0.0.0.0:$PORT app:app"]