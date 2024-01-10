#!/bin/bash -l

#SBATCH --job-name=mlm-ctr
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --mem 32GB
#SBATCH --partition gpu
#SBATCH --gres=gpu:4
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=p.k.hager@uva.nl

source ${HOME}/.bashrc
mamba activate baidu-ultr-features

model="retro-mae-12l-12h"

torchrun --nproc_per_node=4 main.py \
  data=retro-mae \
  model=retro-mae \
  training_arguments.output_dir="output/${model}" \
  +training_arguments.run_name="${model}"
