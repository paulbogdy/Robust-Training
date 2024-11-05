#!/bin/bash -l

#SBATCH --chdir /home/jurcut/Charmer
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 6G
#SBATCH --time 1:30:00
#SBATCH --gres gpu:1
#SBATCH --qos gpu

source ~/venvs/mnlp/bin/activate

model_name=rand_pert_bert_base_sst # Modify the model name to evaluate other models
mv ../Robust-Training/$model_name . 

model_path=./$model_name/model_e5 # Modify the epoch if needed
dataset=sst # Modify the dataset if needed
result_path=results_attack/lm_classifier/basiclm/$dataset

charmer_ending_path=_50iter_encoder_margin_pga0_batch20_1000.csv
final_results_path=$model_name/results

# Create the result path if it does not exist
mkdir -p $final_results_path

# Charmer attacks
charmer_ks=(1 2 10)

for k in ${charmer_ks[@]}; do
    if [[ -f "$final_results_path/charmer_$k.csv" ]]; then
        echo "Charmer $k already exists, skipping..."
    else
        python attack.py \
            --device cuda \
            --loss margin \
            --dataset $dataset \
            --model $model_path \
            --k $k \
            --n_positions 20 \
            --select_pos_mode batch \
            --size 1000 \
            --pga 0

        mv "$result_path/${model_name}_${k}${charmer_ending_path}" "$final_results_path/charmer_$k.csv"
    fi
done

# Other attacks
other_attacks=(textfooler deepwordbug)

for attack in ${other_attacks[@]}; do
    if [[ -f "$final_results_path/$attack.csv" ]]; then
        echo "$attack already exists, skipping..."
    else
        python attack.py \
            --device cuda \
            --loss margin \
            --dataset $dataset \
            --model $model_path \
            --attack_name $attack \
            --size 1000 \
            --pga 0

        mv "$result_path/${attack}_${dataset}_${model_name}.csv" "$final_results_path/$attack.csv"
    fi
done

# Aggregate the results

python ../Robust-Training/evaluate.py \
    --folder_path $final_results_path
