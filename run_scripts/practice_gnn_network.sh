#!/bin/bash
set -e 
ENV_NAME="constraint_graph"  
DATASET_PATH="/projects/anva7450/constraint_graph"
PYTHON_SCRIPT="practice_gnn_network.py"

# activate virtual environment 
echo "Activating venv..." 
module load anaconda 2>/dev/null
conda activate "$ENV_NAME"


# Check if the activation was successful
if [ $? -eq 0 ]; then 
    # Run the Python script
    echo "Running Python script: $PYTHON_SCRIPT"
    export WANDB_API_KEY="47175ab066b28914b78e277766edaaa83d6f4eca"
    python "$PYTHON_SCRIPT" --dataset_pth="$DATASET_PATH"
else
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

# finish and exit 
echo "Deactivating virtual environment..."
conda deactivate
echo "Script finished."