#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Robust-Training
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 12G
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --qos gpu

source ~/venvs/adv-train/bin/activate
python train.py \
    --model_name bert \
    --dataset_name sst \
    --seed 0 \
    --training_method label_contrastive \
    --num_epochs 4 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --q 15 \
    --alpha 0.5 \
    --temperature 0.07 \
    --projection_dim 300 \