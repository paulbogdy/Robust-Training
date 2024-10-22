#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Robust-Training
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 6:00:00
#SBATCH --gres gpu:1
#SBATCH --qos gpu

source ~/venvs/adv-train/bin/activate
python train.py --model_name bert-base-uncased --dataset_name sst --batch_size 64 --learning_rate 2e-4 --alpha 1e-3 --beta 0.5 --num_epochs 5 --atack_iters 10 --save_dir cta_bert_base_sst