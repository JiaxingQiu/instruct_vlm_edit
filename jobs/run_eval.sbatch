#!/bin/bash
#SBATCH --job-name=vlm_edit
#SBATCH --output=logs/vlm_edit_%j.out
#SBATCH --error=logs/vlm_edit_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4

# Activate conda environment
source ~/.bashrc
conda activate revlm  # Adjust environment name as needed

# Set working directory
cd /scratch/jq2uw/MME/instruct_vlm_edit

# Create logs directory if it doesn't exist
mkdir -p logs

# Example command - adjust parameters based on your needs
python -m revlm.eval.py  --model_name llava --dataset_name fvqa --split test --task mcq
python -m revlm.eval.py  --model_name llava --dataset_name fvqa --split test --task mcq --rationale 

python -m revlm.eval.py  --model_name llava --dataset_name aokvqa --split test --task mcq
python -m revlm.eval.py  --model_name llava --dataset_name aokvqa --split test --task mcq --rationale 


