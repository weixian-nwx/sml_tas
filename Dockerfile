FROM python:3.7-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and set working directory
WORKDIR /app

# Install system dependencies (including libGL for OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose the port (Render sets this with $PORT)
EXPOSE 5000

# Start Gunicorn with Eventlet worker (REQUIRED for Flask-SocketIO)
CMD ["gunicorn", "app:app", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:$PORT"]
