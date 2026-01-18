#!/bin/bash

echo "========================================"
echo "VietFood Detection Application"
echo "========================================"
echo ""

# Activate virtual environment if exists
if [ -f venv/bin/activate ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the application
echo "Starting application..."
python main.py
