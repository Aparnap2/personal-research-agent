#!/bin/bash

# Start backend in a subshell
(
    echo "Starting backend..."
    cd /home/aparna/Desktop/personal-research-agent/backend
    source env/bin/activate
    python3 app_research.py
) &

# Start frontend in another subshell
(
    echo "Starting frontend..."
    cd /home/aparna/Desktop/personal-research-agent/frontend
    pnpm run dev
) &

# Wait for both processes and clean up
wait
trap 'kill $(jobs -p)' EXIT