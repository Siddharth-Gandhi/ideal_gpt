#!/bin/bash
#SBATCH --job-name=j
#SBATCH --partition=gpu
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs/jupyter.log
#SBATCH --nodelist=boston-2-35

source ~/miniconda3/etc/profile.d/conda.sh
conda activate idl

# cat /etc/hosts
node=$(hostname -s)
port=8889

jupyter-lab --port=${port} --ip=${node} --no-browser