#!/bin/bash
#SBATCH --job-name=gnn-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=anva7450@colorado.edu
#SBATCH --partition=amilan
#SBATCH --qos=normal

module purge
module load intel
module load mkl

echo "== Starting job =="
sleep 1

# Call your bash script that activates conda and runs the NN
bash practice_gnn_network.sh

echo "== End of Job =="