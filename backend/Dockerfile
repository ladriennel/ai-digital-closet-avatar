# Use platform-specific base image for better compatibility
FROM --platform=linux/arm64 python:3.11-slim

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose the port the Flask app will run on
EXPOSE 5000

# Set the default command to run your Flask app
CMD ["python", "run.py"]
