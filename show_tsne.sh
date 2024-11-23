#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Charmer
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 6G
#SBATCH --time 10:00
#SBATCH --gres gpu:1
#SBATCH --qos gpu

source ~/venvs/mnlp/bin/activate

model_path=./$model_name # Modify the epoch if needed

python show_tsne.py --model_path $model_path