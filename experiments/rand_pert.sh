#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Robust-Training
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 6:00:00
#SBATCH --gres gpu:1
#SBATCH --qos gpu

source ~/venvs/adv-train/bin/activate
python train.py --model_name bert-base-uncased --training_method random_pert --dataset_name sst --batch_size 64 --learning_rate 2e-4 --q 0.05 --num_epochs 5 --save_dir rand_pert_bert_base_sst