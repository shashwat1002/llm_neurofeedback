"""vLLM version of main_predexp.py

Precomputes all ICL prompts across all experiments, then runs a single batched
vLLM inference call for logits and writes per-experiment cache files.

Usage:
    python main_predexp_vllm.py --config_s llama3_3b --config_e llama3_3b \
        --dataset commonsense --tp 4
"""

import os
from pathlib import Path
from collections import defaultdict
from itertools import groupby
import platform
import numpy as np
import pandas as pd
import torch
from joblib import load
import argparse

from neurofeedback import _init_neurofeedback_prompts, generate_ICL_examples
from utils import seed_everything, load_exp_cfg, load_labeler, safe_dump, load_yaml, find_tags_indices


def build_save_indices(n_examples):
    if n_examples <= 200:
        return list(range(n_examples))
    return list(range(201)) + list(range(210, n_examples, 10))


def _get_cache_dir(cfg_yaml):
    if platform.system() == 'Linux' and 'gatech' in platform.node():
        return f'{cfg_yaml["cache_dir"]}/downloaded_models'
    return './downloaded_models'


def precompute_all_prompts(examples_scores, labeler, cfg, cfg_e, f_name, exp_save_dir, tokenizer):
    """Precompute all (experiment × flip × layer) prompts for all pending experiments.

    Returns a list of entry dicts and a dict mapping pred_exp → sampled index list.
    """
    prompt_cfg, meta_prompt = _init_neurofeedback_prompts(labeler, 'report_NF')

    entries = []
    exp_sample_idx = {}

    for pred_exp in range(cfg_e.exp_id_start, cfg_e.exp_id_end):
        file_name = f"predict_hidden_{cfg.process_hidden_method}_clf_{f_name}_exp{pred_exp}.pkl"
        if os.path.exists(exp_save_dir / file_name):
            print(f"Experiment {pred_exp} already completed, skipping.")
            continue

        seed_everything(42 + pred_exp)
        examples_scores_exp = examples_scores.sample(cfg_e.n_icl_examples_report, random_state=42 + pred_exp)
        exp_sample_idx[pred_exp] = examples_scores_exp.index.tolist()

        n_examples = len(examples_scores_exp)
        all_layers = sorted(col for col in examples_scores_exp.columns if not isinstance(col, str))
        sentences = examples_scores_exp['sentences'].tolist()

        for flip_shown_label in [False, True]:
            for layer in all_layers:
                labeler.fit(examples_scores_exp[layer].to_numpy())
                prompts_icl, label_scores, original_scores = generate_ICL_examples(
                    prompt_cfg["user_msg"], sentences,
                    examples_scores_exp[layer].tolist(),
                    n_examples, labeler, flip_shown_label,
                )
                prompt_str = tokenizer.apply_chat_template(
                    meta_prompt + prompts_icl, tokenize=False
                )
                entries.append({
                    'pred_exp': pred_exp,
                    'flip': flip_shown_label,
                    'layer': layer,
                    'prompt_str': prompt_str,
                    'label_scores': label_scores,
                    'original_scores': original_scores,
                    'file_name': file_name,
                })

    return entries, exp_sample_idx


def run_vllm_logits(llm, prompt_strings, n_logprobs=200):
    """Batch inference via vLLM. Returns prompt_logprobs for every prompt."""
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=n_logprobs,
        temperature=0,
        seed=42,
    )
    return llm.generate(prompt_strings, sampling_params, use_tqdm=True)


def assemble_and_save(entries, vllm_outputs, labeler, save_indices, exp_save_dir,
                      min_score, possible_choices, choice_token_ids,
                      text_precede_label, text_precede_label_tokens):
    """Assemble per-experiment DataFrames and write pkl files."""
    eps = 1e-8

    combined = list(zip(entries, vllm_outputs))
    combined.sort(key=lambda x: x[0]['pred_exp'])

    for pred_exp, group_iter in groupby(combined, key=lambda x: x[0]['pred_exp']):
        group = list(group_iter)
        file_name = group[0][0]['file_name']

        est_score_dt = defaultdict(list)
        layers_scores_dt = {'layer': [], 'original_scores': [], 'labeler': [], 'label_scores': []}

        for entry, vllm_out in group:
            flip = entry['flip']
            layer = entry['layer']
            label_scores = entry['label_scores']
            original_scores = entry['original_scores']

            prompt_token_ids = torch.tensor(vllm_out.prompt_token_ids)
            occurrences = find_tags_indices(
                prompt_token_ids,
                [(text_precede_label, text_precede_label_tokens)]
            )[text_precede_label]

            prompt_logprobs = vllm_out.prompt_logprobs  # list[None | dict[int, Logprob]]

            all_example_est_probs = []
            for _start_idx, end_idx in occurrences:
                # prompt_logprobs[end_idx] is the distribution predicting the token at end_idx,
                # equivalent to logits[:, end_idx-1, :] in HF
                pos_lp = prompt_logprobs[end_idx]
                if pos_lp is None:
                    probs = torch.ones(len(possible_choices)) / len(possible_choices)
                else:
                    lps = torch.tensor([
                        pos_lp[tid].logprob if tid in pos_lp else -100.0
                        for tid in choice_token_ids
                    ])
                    probs = torch.softmax(lps, dim=-1)
                all_example_est_probs.append(probs)

            all_example_est_probs = torch.stack(all_example_est_probs, dim=0)
            all_example_est_scores = (all_example_est_probs.argmax(dim=1) + min_score).tolist()
            logit_diff = torch.log(
                (all_example_est_probs[:, -1] + eps) / (all_example_est_probs[:, 0] + eps)
            ).tolist()

            est_score_dt['layer'].append(layer)
            est_score_dt['flip_shown_label'].append(flip)
            est_score_dt['all_example_est_scores'].append(all_example_est_scores)
            est_score_dt['all_example_true_scores'].append(label_scores)
            est_score_dt['all_example_est_scores_logitdiff'].append(logit_diff)

            if not flip:
                layers_scores_dt['layer'].append(layer)
                layers_scores_dt['original_scores'].append(original_scores)
                layers_scores_dt['label_scores'].append(label_scores)
                layers_scores_dt['labeler'].append(labeler)

        safe_dump(pd.DataFrame(layers_scores_dt), exp_save_dir / f"original_{file_name}")
        safe_dump(pd.DataFrame(est_score_dt), exp_save_dir / file_name)
        print(f"Experiment {pred_exp} saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Setup (vLLM version)")
    parser.add_argument("--config_s", type=str, default="llama3_3b")
    parser.add_argument("--config_e", type=str, default="llama3_3b")
    parser.add_argument("--dataset", type=str, default="commonsense")
    parser.add_argument("--clf", type=str, default="default")
    parser.add_argument("--pc", type=int, default=1)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--randomize", action='store_true')
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (number of GPUs for vLLM)")
    parser.add_argument("--n_logprobs", type=int, default=200, help="Top-K logprobs per position from vLLM")
    parser.add_argument("--max_model_len", type=int, default=32768, help="Max sequence length for KV cache allocation")
    args = parser.parse_args()

    cfg = load_exp_cfg(args.config_s, pc_number=args.pc, clf=args.clf, randomize=args.randomize)
    cfg_e = load_exp_cfg(args.config_e)
    seed_everything(42)

    hiddens_save_dir = Path("results") / cfg.model_name.replace("/", "_") / args.dataset
    f_name = cfg.clf if cfg.clf == "lr" else f'{cfg.clf}_pc{cfg.pc_number}'
    if args.randomize:
        f_name += "_randomized"
    examples_scores = load(hiddens_save_dir / f"hidden_{cfg.process_hidden_method}_{f_name}_example_scores.pkl")

    save_dir = Path("results") / (cfg.model_name.replace("/", "_") + '-' + cfg_e.model_name.replace("/", "_")) / args.dataset
    os.makedirs(save_dir, exist_ok=True)
    labeler, exp_save_dir = load_labeler(args.scale, cfg.quantile_transform, save_dir)
    save_indices = build_save_indices(cfg_e.n_icl_examples_report)

    cfg_yaml = load_yaml(Path("configs") / "nf_exp1.yml")
    cache_dir = _get_cache_dir(cfg_yaml)

    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    from vllm import LLM
    llm = LLM(
        model=cfg_e.model_name,
        tensor_parallel_size=args.tp,
        dtype="float16",
        download_dir=cache_dir,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()

    min_score, max_score = labeler.min_points, labeler.max_points
    possible_choices = [str(i) for i in range(min_score, max_score + 1)]
    choice_token_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in possible_choices]

    text_precede_label = "Score: "
    text_precede_label_tokens = tokenizer(
        text_precede_label, return_tensors="pt", add_special_tokens=False
    )['input_ids'][0]

    # === Phase 1: Precompute all prompts ===
    print("Precomputing all ICL prompts...")
    entries, exp_sample_idx = precompute_all_prompts(
        examples_scores, labeler, cfg, cfg_e, f_name, exp_save_dir, tokenizer
    )

    if not entries:
        print("All experiments already completed.")
        exit(0)

    print(f"Total prompts to process: {len(entries)} "
          f"({len(exp_sample_idx)} experiments × layers × 2 flip values)")

    # === Phase 2: vLLM batch logit inference (all prompts at once) ===
    print("Running vLLM inference...")
    vllm_outputs = run_vllm_logits(llm, [e['prompt_str'] for e in entries], n_logprobs=args.n_logprobs)

    # === Phase 3: Assemble and save per-experiment cache files ===
    print("Saving results...")
    assemble_and_save(
        entries, vllm_outputs, labeler, save_indices, exp_save_dir,
        min_score, possible_choices, choice_token_ids,
        text_precede_label, text_precede_label_tokens,
    )

    idx_array = np.array([exp_sample_idx[e] for e in sorted(exp_sample_idx)])
    suffix = "_randomized" if args.randomize else ""
    np.savez_compressed(
        exp_save_dir / f"predict_exp_examples_idx{suffix}.npz",
        examples_idx=idx_array,
    )
    np.savez_compressed(
        exp_save_dir / f"predict_exp_save_indices{suffix}.npz",
        save_indices=np.array(save_indices),
    )
    print("Done.")
