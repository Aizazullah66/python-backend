FROM python:3.12-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY server.py .

# Expose the port (Railway assigns PORT dynamically)
EXPOSE 5000

# Command to run the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]