#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multi_312


python main_prep.py --model llama3_70b --dataset commonsense 
python main_prep.py --model llama3_70b --dataset commonsense --randomize

