#!/bin/bash

# Get the prefix from the first argument, default to empty if not provided
PREFIX="${1:-}"


# Define the seeds for the repetitions
# SEEDS=(27 42 404 1312 1984)
SEEDS=(42 404 1312)

# Activate the virtual environment
. .venv/bin/activate

for SEED in "${SEEDS[@]}"; do
    # Define the filename with the prefix and seed
    FILENAME="${PREFIX}_${SEED}_training.out"

    # Create or clear the file
    > "$FILENAME"

    echo "Logging output to $FILENAME"

    # Start the Python script with nohup and redirect output to the file
    # Pass the current seed and other arguments
    nohup nice -n 10 python3 train_model.py "${PREFIX}_${SEED}" --seed "$SEED" "${@:2}" > "$FILENAME" 2>&1 &    PID=$!
    echo "Python script for seed $SEED started with PID: $PID"
    echo "PID: $PID" >> "$FILENAME"
    # Set sleep to avoid duplicated automatic GPU assignment for multiple seeds
    sleep 30
done

echo "All training runs started."
echo "You can monitor the logs with: tail -f seed_${PREFIX}_training.out"