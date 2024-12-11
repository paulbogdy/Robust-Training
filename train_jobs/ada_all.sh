#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Robust-Training
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 2:00:00
#SBATCH --gres gpu:1
#SBATCH --qos gpu

source ~/venvs/adv-train/bin/activate
python train.py \
    --model_name bert \
    --dataset_name sst \
    --seed 0 \
    --training_method base \
    --num_epochs 4 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --aug_folder data/ada \
    --aug_type all 