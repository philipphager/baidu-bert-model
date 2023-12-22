#!/bin/bash -l

#SBATCH --job-name=baidu-bert
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64GB
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=p.k.hager@uva.nl

source ${HOME}/.bashrc
mamba activate baidu-bert-model

python main.py
