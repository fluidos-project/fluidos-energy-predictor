#!/bin/bash

# Run the Python script with some arguments
output=$(python3 main.py --model test --curve test --epochs 1 2>&1)

# Check if the command was successful
if [[ $output == *"No folders found in"* ]]; then
    echo "Python script runs correctly up to the first option."
else
    echo "Python script does not run correctly. Error: $output"
fi