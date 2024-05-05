# Use an official Python runtime as a parent image
FROM python:3.10-slim-bookworm

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    musl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python3 -m pip cache purge
RUN python3 -m pip install --no-cache-dir -r requirements.txt


# Expose the port that the application will run on
EXPOSE 7860

# Set the entrypoint with a default command and allow the user to override it
ENTRYPOINT ["python3", "chat.py"]