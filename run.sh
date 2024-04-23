#!/bin/bash
#SBATCH --job-name=pv2
#SBATCH --partition=gpu
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs/jupyter.log
#SBATCH --nodelist=boston-1-7

source ~/miniconda3/etc/profile.d/conda.sh
conda activate idl

# cat /etc/hosts

nvidia-smi

echo "starting run"
python transformer_v2.py
echo "job done"