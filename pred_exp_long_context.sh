#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multi_312

torchrun --nproc_per_node=2 main_predexp.py --config_s qwen2.5_7b_1m  --config_e qwen2.5_7b_1m  --clf lr --dataset sst2 --no-hiddens







# torchrun --nproc_per_node=2 main_predexp.py --config_s qwen2.5_7b_1m  --config_e qwen2.5_7b_1m  --clf lr --dataset commonsense --no-hiddens


# torchrun --nproc_per_node=2 main_predexp.py --config_s qwen2.5_7b_1m  --config_e qwen2.5_7b_1m  --clf lr --dataset commonsense --randomize --no-hiddens