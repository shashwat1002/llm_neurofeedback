"""Given n ICL examples (sentences, neurofeedback scores), predict the neurofeedback scores for new examples"""

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
# from functools import partialmethod

# VERBOSE = False
# if not VERBOSE:
#     tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from joblib import load
import argparse
from neurofeedback import predict_score_by_examples, predict_score_by_examples_batched, predict_score_by_examples_tensor_parallel
from utils import seed_everything, load_lm, load_exp_cfg, load_labeler, load_lm_tp_parallel
import torch.distributed as dist
import torch

def build_save_indices(n_examples):
    if n_examples <= 200:
        return list(range(n_examples))
    return list(range(201)) + list(range(210, n_examples, 10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Setup")
    parser.add_argument("--config_s", type=str, default="llama3_3b", help="Config file: model that generates scores")
    parser.add_argument("--config_e", type=str, default="llama3_3b", help="Config file: model that runs experiments")
    parser.add_argument("--dataset", type=str, default="commonsense")  # commonsense, true_false, sycophancy
    parser.add_argument("--clf", type=str, default="default")  # default classifier in loaded cfg
    parser.add_argument("--pc", type=int, default=1)
    parser.add_argument("--scale", type=int, default=2, help="Score scale: 2 for binary, >2 for Likert scale.")
    parser.add_argument("--randomize", action='store_true', help="whether to randomize the labels as a control")
    parser.add_argument("--quantize", action='store_true', help="load the experiment model in 4-bit (bitsandbytes)")
    parser.add_argument("--no-hiddens", action='store_true', help="skip saving processed_hiddens to reduce memory usage")
    args = parser.parse_args()
    cfg = load_exp_cfg(args.config_s, pc_number=args.pc, clf=args.clf, randomize=args.randomize)  # pc of subject model's hiddens
    cfg_e = load_exp_cfg(args.config_e)  # model to understand and predict scores
    seed_everything(42)

    if not args.quantize:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    hiddens_save_dir = Path("results") / cfg.model_name.replace("/", "_") / args.dataset
    if cfg.clf == "lr":
        f_name = cfg.clf
    else:
        f_name = f'{cfg.clf}_pc{cfg.pc_number}'
    if args.randomize:
        f_name += "_randomized"
    examples_scores = load(hiddens_save_dir / f"hidden_{cfg.process_hidden_method}_{f_name}_example_scores.pkl")

    save_dir = Path("results") / (cfg.model_name.replace("/", "_") + '-'+ cfg_e.model_name.replace("/", "_")) / args.dataset
    os.makedirs(save_dir, exist_ok=True)
    model, tokenizer = load_lm_tp_parallel(cfg_e.model_name, quantize=args.quantize)
    labeler, exp_save_dir = load_labeler(args.scale, cfg.quantile_transform, save_dir)
    save_indices = build_save_indices(cfg_e.n_icl_examples_report)

    exp_examples_idx = []
    for pred_exp in trange(cfg_e.exp_id_start, cfg_e.exp_id_end, desc="Predict experiment"):
        file_name = f"predict_hidden_{cfg.process_hidden_method}_clf_{f_name}_exp{pred_exp}.pkl"
        if os.path.exists(exp_save_dir / file_name):
            print(f"Experiment {pred_exp} already completed.")
            continue
        seed_everything(42 + pred_exp)
        examples_scores_exp = examples_scores.sample(cfg_e.n_icl_examples_report, random_state=42 + pred_exp)
        exp_examples_idx.append(examples_scores_exp.index.tolist())
        save_hiddens = not args.no_hiddens
        if args.quantize:
            predict_score_by_examples_batched(model, tokenizer, examples_scores_exp, labeler, cfg.process_hidden_method,
                                              exp_save_dir / file_name, save_indices, batch_size=1, save_hiddens=save_hiddens)
        else:
            predict_score_by_examples_tensor_parallel(model, tokenizer, examples_scores_exp, labeler, cfg.process_hidden_method,
                                                      exp_save_dir / file_name, save_indices, save_hiddens=save_hiddens)
        print(f"Experiment {pred_exp} completed and saved.")
    if not args.randomize:
        np.savez_compressed(exp_save_dir / "predict_exp_examples_idx.npz", examples_idx=np.array(exp_examples_idx))  # shape: (n_experiments, n_icl_examples)
        np.savez_compressed(exp_save_dir / "predict_exp_save_indices.npz", save_indices=np.array(save_indices))
    else:
        np.savez_compressed(exp_save_dir / "predict_exp_examples_idx_randomized.npz", examples_idx=np.array(exp_examples_idx))  # shape: (n_experiments, n_icl_examples)
        np.savez_compressed(exp_save_dir / "predict_exp_save_indices_randomized.npz", save_indices=np.array(save_indices))