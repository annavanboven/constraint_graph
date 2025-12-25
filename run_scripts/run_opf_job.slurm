#!/bin/bash
#SBATCH --job-name=gnn-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=anva7450@colorado.edu
#SBATCH --partition=amilan
#SBATCH --qos=normal

module purge
module load intel
module load mkl

set -e 
CASE_NAME="case6ww"
ENV_NAME="constraint_graph"  
PYTHON_SCRIPT="train_pdgnn"

# activate virtual environment 
echo "Activating venv..." 
module load anaconda 2>/dev/null
conda activate "$ENV_NAME"

# activate wandb 
echo "Activating wandb..."
export WANDB_API_KEY="47175ab066b28914b78e277766edaaa83d6f4eca"
SWEEP_ID=$(python create_opf_sweep.py)

echo "== Starting job =="
sleep 1

# Call  python script
if [ $? -eq 0 ]; then 
    # Run the Python script
    echo "Running Python script: $PYTHON_SCRIPT"
    srun --mpi=none python "$PYTHON_SCRIPT" --project="$CASE_NAME" --sweep_id="$SWEEP_ID"
else
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

# finish and exit 
echo "Deactivating virtual environment..."
conda deactivate
echo "== End of Job =="