#!/bin/bash

# Install dependencies
# pip install -r requirements.txt

# Run the web server
echo "Starting web server on http://localhost:9000"
uvicorn main:app --host 0.0.0.0 --port 9000 --reload