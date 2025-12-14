#!/bin/bash
# Helper script to activate the reporter conda environment
# Usage: source activate_env.sh

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate the reporter environment
conda activate reporter

if [ $? -eq 0 ]; then
    echo "✅ Activated 'reporter' conda environment"
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
else
    echo "❌ Failed to activate 'reporter' conda environment"
    echo "Please create it first with: conda create -n reporter python=3.10"
    exit 1
fi

