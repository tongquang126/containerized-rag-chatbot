# Use a lightweight Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock first to install dependencies
COPY Pipfile Pipfile.lock ./

# Install dependencies via Pipenv
RUN pipenv install --system --deploy

# Copy rest of the code
COPY . .

# Expose the port Flask app runs on
EXPOSE 5001

# Run using pipenv environment
CMD ["pipenv", "run", "python", "src/app.py"]
