# Smart Traffic Congestion Forecasting System — Unified Docker Blueprint
FROM python:3.11-slim

# Install Node.js & NPM for building frontend
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements & install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend & build assets
COPY frontend/package*.json ./frontend/
RUN npm --prefix frontend install
COPY frontend/ ./frontend/
RUN npm --prefix frontend run build

# Copy remaining backend and project files
COPY backend/ ./backend/
COPY data/ ./data/
COPY models/ ./models/
COPY scripts/ ./scripts/

ENV PORT=8000
ENV DEMO_MODE=true
ENV BACKEND_HOST=0.0.0.0

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
