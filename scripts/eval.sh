#!/bin/bash -l

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Check if arguments were provided:
if [ -z "$1" ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

model=$1

source ${HOME}/.bashrc
mamba activate baidu-bert-model

python eval.py data=base model=$model $2
