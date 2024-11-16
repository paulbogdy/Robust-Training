#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Robust-Training
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --qos gpu

source ~/venvs/adv-train/bin/activate
python train.py \
    --model_name bert \
    --dataset_name sst \
    --seed 0 \
    --training_method rand_char_v4 \
    --num_epochs 4 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --q 10 \
    --insertion_rate 1 \
    --deletion_rate 0 \
    --alph_dist uniform \
    --pos_dist uniform \