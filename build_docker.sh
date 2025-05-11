#!/bin/bash

# Build and start the Docker containers for the research agent

echo "Building Docker containers..."
docker-compose build

echo "Starting Docker containers..."
docker-compose up -d

echo "Docker containers are now running."
echo "You can view logs with: docker-compose logs -f"
echo "You can stop containers with: docker-compose down"