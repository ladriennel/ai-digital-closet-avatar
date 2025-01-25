FROM python:3.11-slim

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
