#!/bin/bash
#SBATCH --job-name=gemma_trial
#SBATCH --output=logs/gemma_%j.out
#SBATCH --error=logs/gemma_%j.err
#SBATCH --partition=dsba
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00

# Load CUDA module
module load nvidia/cuda-12.4.0

# Set up environment
export PATH=$HOME/bin:$PATH
export LD_LIBRARY_PATH=$HOME/lib/ollama/cuda_v12:$LD_LIBRARY_PATH

# Activate conda
source /opt/share/modulefiles/sw/miniconda3/etc/profile.d/conda.sh
conda activate nlp_env

# Start Ollama server in background
ollama serve > ollama_server.log 2>&1 &
sleep 10  # wait for server to be ready

# Loop through models listed in models.txt
while IFS= read -r model; do
    echo "Pulling model: $model"
    ollama pull "$model"

    echo "Running trial.py with model: $model"
    python trial1.py "$model"

    echo "Removing model: $model"
    ollama rm "$model"

    echo "Done with model: $model"
    echo "-----------------------------"
done < models.txt

conda deactivate

