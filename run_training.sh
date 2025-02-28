#!/bin/bash

# Get the prefix from the first argument, default to empty if not provided
PREFIX="$1"

# Define the filename with the prefix (if any)
FILENAME="${PREFIX}_file.out"

# Create or clear the file
> "$FILENAME"

echo "Logging output to $FILENAME"

# Start the Python script with nohup and redirect output to the file
. .venv/bin/activate
nohup python3 train_model.py > "$FILENAME" 2>&1 &
PID=$!
echo "Python script started with PID: $PID"
echo "PID: $PID" >> "$FILENAME"

sleep 1
tail -f "$FILENAME"