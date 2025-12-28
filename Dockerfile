FROM python:3.10-slim

# Avoid Python buffering + speed up
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# System deps (needed for xgboost, numpy, tensorflow)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Create output directories if they don't exist
RUN mkdir -p energy_load_next/models mlruns

# Flexible entrypoint - users pass args at runtime
CMD ["sleep", "infinity"]
