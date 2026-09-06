#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_312

python probe_hidden.py \
    --pkl_path results/meta-llama_Llama-3.1-70B-Instruct/commonsense/hidden_mean_pcascore_pc1_example_scores.pkl \
    --layer 0 \
    --n_jobs 2

python probe_hidden.py \
    --pkl_path results/meta-llama_Llama-3.1-70B-Instruct/commonsense/hidden_mean_pcascore_pc2_example_scores.pkl \
    --layer 0 \
    --n_jobs 2

python probe_hidden.py \
    --pkl_path results/meta-llama_Llama-3.1-70B-Instruct/commonsense/hidden_mean_pcascore_pc4_example_scores.pkl \
    --layer 0 \
    --n_jobs 2

python probe_hidden.py \
    --pkl_path results/meta-llama_Llama-3.1-70B-Instruct/commonsense/hidden_mean_pcascore_pc8_example_scores.pkl \
    --layer 0 \
    --n_jobs 2

python probe_hidden.py \
    --pkl_path results/meta-llama_Llama-3.1-70B-Instruct/commonsense/hidden_mean_pcascore_pc8_example_scores.pkl \
    --layer 0 \
    --n_jobs 2

python probe_hidden.py \
    --pkl_path results/meta-llama_Llama-3.1-70B-Instruct/commonsense/hidden_mean_pcascore_pc32_example_scores.pkl \
    --layer 0 \
    --n_jobs 2


python probe_hidden.py \
    --pkl_path results/meta-llama_Llama-3.1-70B-Instruct/commonsense/hidden_mean_pcascore_pc512_example_scores.pkl \
    --layer 0 \
    --n_jobs 2
