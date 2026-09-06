#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multi_312

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


torchrun --nproc_per_node=2 main_predexp.py --config_s llama3.1_70b --config_e llama3.1_70b --clf pcascore --pc 1 --dataset commonsense
# torchrun --nproc_per_node=2 main_predexp.py --config_s llama3.1_70b --config_e llama3.1_70b --clf pcascore --pc 2 --dataset commonsense
# torchrun --nproc_per_node=2 main_predexp.py --config_s llama3.1_70b --config_e llama3.1_70b --clf pcascore --pc 4 --dataset commonsense
# torchrun --nproc_per_node=2 main_predexp.py --config_s llama3.1_70b --config_e llama3.1_70b --clf pcascore --pc 8 --dataset commonsense
# torchrun --nproc_per_node=2 main_predexp.py --config_s llama3.1_70b --config_e llama3.1_70b --clf pcascore --pc 32 --dataset commonsense


# python main_predexp.py --config_s llama3.1_70b --config_e llama3.1_70b --clf pcascore --pc 128 --dataset commonsense
# python main_predexp.py --config_s llama3.1_70b --config_e llama3.1_70b --clf pcascore --pc 512 --dataset commonsense
