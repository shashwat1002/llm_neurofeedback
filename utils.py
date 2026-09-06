import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union, List
import platform
import numpy as np
import pandas as pd
import torch
from joblib import dump, load, Parallel, delayed
from ruamel.yaml import YAML
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

MODEL_ALIASES = {
    'llama3.1_70b': 'meta-llama/Llama-3.1-70B-Instruct',
    'llama3_70b': 'meta-llama/Llama-3.1-70B-Instruct',
    'llama3.1_8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama3_8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama3.2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'llama3_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'llama3.2_3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama3_3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'qwen2.5_72b': 'Qwen/Qwen2.5-72B-Instruct',
    'qwen2.5_7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5_1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5_3b': 'Qwen/Qwen2.5-3B-Instruct',
    'qwen2.5_7b_1m': 'Qwen/Qwen2.5-7B-Instruct-1M',
    'qwen2.5_14b_1m': 'Qwen/Qwen2.5-14B-Instruct-1M',
}

LARGE_MODEL_ALIASES = {'llama3.1_70b', 'llama3_70b', 'qwen2.5_72b'}


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_yaml(file_path: Union[Path, str]) -> Dict:
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(file_path, 'r', encoding='utf-8') as file:
        yaml_dict = yaml.load(file)
    return yaml_dict


def load_exp_cfg(model_name: str, pc_number: int = 3, clf='default', randomize=False):
    ROOT = Path(__file__).resolve().parents[0]
    cfg = load_yaml(ROOT / "configs" / "nf_exp1.yml")

    cfg['model_name'] = MODEL_ALIASES.get(model_name, model_name)
    cfg['randomize'] = randomize
    if clf == 'default':
        clf = cfg['clf']
    else:
        cfg['clf'] = clf

    if clf in ['pcascore','pcadiff']:
        cfg['pc_number'] = pc_number
        cfg['clf_name'] = f'pc{pc_number}'
    elif clf in ['lr']:
        cfg['pc_number'] = None
        cfg['clf_name'] = clf
    else:
        raise ValueError(f"Unknown classifier: {clf}")

    cfg['n_train_examples'] = cfg['n_icl_examples_control']
    if model_name in LARGE_MODEL_ALIASES:
        cfg['n_train_examples'] = [256]

    cfg = SimpleNamespace(**cfg)
    return cfg


def load_lm(model_name_or_path, device=None):
    cfg_path = Path(__file__).resolve().parent / "configs" / "nf_exp1.yml"
    cfg = load_yaml(cfg_path)
    padding_side = cfg.get("padding_side", "left")  # should be 'left'
    dtype = cfg.get("dtype", "float16")
    if platform.system() == 'Linux' and 'gatech' in platform.node():  # for pace hpc
        hpc_cache_dir = cfg['cache_dir']
        cache_dir = f'{hpc_cache_dir}/downloaded_models'
    else:
        cache_dir = './downloaded_models'
    print("Loading model:", model_name_or_path, 'from:', cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map="auto", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side, pad_to_multiple_of=8)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    print('Padding side is set to', tokenizer.padding_side)
    torch.set_grad_enabled(False)
    seed_everything(42)
    return model, tokenizer


def load_lm_tp_parallel(model_name_or_path, quantize=False):
    """Load a model for multi-GPU inference.

    Without --quantize: tensor parallelism via tp_plan="auto". Each rank holds a
    shard of every weight; forward passes run in parallel across all ranks with an
    all-reduce per layer. Requires torchrun and an initialised process group.

    With --quantize: pipeline parallelism via device_map="auto" with 4-bit
    bitsandbytes quantization. Layers are distributed across GPUs sequentially.
    Run with plain python (single process); no torchrun needed.
    """
    import torch.distributed as dist

    cfg_path = Path(__file__).resolve().parent / "configs" / "nf_exp1.yml"
    cfg = load_yaml(cfg_path)
    padding_side = cfg.get("padding_side", "left")
    dtype = cfg.get("dtype", "float16")
    if platform.system() == 'Linux' and 'gatech' in platform.node():
        cache_dir = f'{cfg["cache_dir"]}/downloaded_models'
    else:
        cache_dir = './downloaded_models'
    os.makedirs(cache_dir, exist_ok=True)

    if quantize:
        from transformers import BitsAndBytesConfig
        n_gpus = torch.cuda.device_count()
        mem_per_gpu = f"{int(torch.cuda.get_device_properties(0).total_memory * 0.9 / 1024**3)}GiB"
        max_memory = {i: mem_per_gpu for i in range(n_gpus)}
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        print(f"Loading model (pipeline parallel, 4-bit quantized) across {n_gpus} GPUs ({mem_per_gpu} each):", model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            max_memory=max_memory,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            cache_dir=cache_dir,
        )
    else:
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialised before load_lm_tp_parallel. "
                "Call dist.init_process_group('nccl') and torch.cuda.set_device(local_rank) first."
            )
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        if local_rank == 0:
            print("Loading model (tensor parallel):", model_name_or_path, 'from:', cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            tp_plan="auto",
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            cache_dir=cache_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side=padding_side, pad_to_multiple_of=8
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    print('Padding side is set to', tokenizer.padding_side)
    torch.set_grad_enabled(False)
    seed_everything(42)
    return model, tokenizer


@torch.inference_mode()
def generate_text(
        model,
        tokenizer,
        prompts,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        keep_new=False,
        batch_size=8,
        verbose=True,
        skip_special_tokens=False,
):
    # If prompt is a string, convert it to a list to support batching.
    is_single = False
    if isinstance(prompts, str):
        prompts = [prompts]
        is_single = True

    total_length = len(prompts)
    # We track a pointer to where we are in `tokens`, and while there's data left, we keep generating.
    i = 0
    outputs = []
    generated_texts = []

    with trange(0, total_length, batch_size, desc="Generate all texts", disable=not verbose) as pbar:
        pbar.reset(total=total_length)  # total steps = total_length

        while i < total_length:
            success = False
            while not success:
                try:
                    current_batch_end = min(i + batch_size, total_length)  # make sure don't exceed total_length

                    gen_inputs = tokenizer(prompts[i:current_batch_end], return_tensors="pt", padding=True).to(model.device)
                    output = model.generate(
                        **gen_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                    )
                    success = True
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    new_batch_size = max(batch_size // 2, 1)  # Reduce batch_size by half
                    if new_batch_size == batch_size:
                        raise RuntimeError("Ran out of memory even with batch_size=1. Cannot proceed.")
                    print(f"OutOfMemoryError. Reducing batch_size from {batch_size} to {new_batch_size} and retrying.")
                    batch_size = new_batch_size
                    # Also update the tqdm bar step since we didn't advance yet
                    pbar.total = total_length
                    pbar.n = i
                    pbar.refresh()

            output = list(output.detach().cpu())
            outputs.extend(output)
            for idx, out in enumerate(output):
                if keep_new:  # Keep only the newly generated tokens
                    input_len = gen_inputs["input_ids"].shape[1]
                    gen_text = tokenizer.decode(out[input_len:], skip_special_tokens=skip_special_tokens)
                else:
                    gen_text = tokenizer.decode(out, skip_special_tokens=skip_special_tokens)
                generated_texts.append(gen_text)

            i = current_batch_end
            pbar.update(current_batch_end - pbar.n)

    assert len(generated_texts) == len(prompts), f"Expected {len(prompts)} generated texts, but got {len(generated_texts)}"
    # If the original input was a single string, return a single string.
    if is_single:
        return generated_texts[0]
    return generated_texts


def safe_dump(obj, file, fmt='lzma', level=3):
    try:
        dump(obj, file, compress=(fmt, level))  # 'lzma', 'lz4'
    except ValueError:
        dump(obj, file, compress=('zlib', 3))
    except FileNotFoundError:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        dump(obj, file, compress=(fmt, level))
    except OSError:
        print(f"[dump error] {file}, renaming to {file}_tmp")
        dump(obj, f'{file}_tmp', compress=(fmt, level))


def parallel_load(file_paths: List[str], n_jobs: int = -1) -> List[object]:
    """Parallel load of multiple joblib files.
    :param file_paths: List of joblib file paths.
    :param n_jobs: Number of parallel workers (-1 = use all cores).
    """
    def safe_load(path):
        try:
            return load(path)
        except Exception as e:
            print(f"[load error] {path}: {e}")
            return None

    def is_debugging():
        if os.getenv('PYCHARM_HOSTED') == '1' and os.getenv('PYDEVD_LOAD_VALUES_ASYNC') is not None:
            return True
        if sys.gettrace() is not None:
            return True
        return False

    actual_n_jobs = 1 if is_debugging() else n_jobs
    results = Parallel(n_jobs=actual_n_jobs)(delayed(safe_load)(path) for path in file_paths)
    return [r for r in results if r is not None]


def load_saved_data(file_name, save_dir, experiment='control', start=0, end=100, verbose=False):
    n_exp = 0
    file_paths = []
    for i_exp in range(start, end):
        name = f"{experiment}_{file_name}_exp{i_exp}.pkl"
        if os.path.exists(f"{save_dir}/{name}"):
            if verbose:
                print(f"load {save_dir}/{name}")
            file_paths.append(f"{save_dir}/{name}")
            n_exp += 1
        else:
            if verbose:
                print(f"Experiment {save_dir}/{name} not found.")
            continue
    all_scores = parallel_load(file_paths)
    for i, df in enumerate(all_scores):
        df['experiment'] = i
    if len(all_scores) == 0:
        raise ValueError(f"No experiments found for {experiment}_{file_name} at {save_dir}.")
    print(f"Loaded {n_exp}/{end - start} experiments for {experiment}_{file_name} at {save_dir}.")
    all_scores = pd.concat(all_scores, ignore_index=True)
    return all_scores


def extract_assistant_header(tokenizer, tokenize=False):
    msg = [{"role": "user", "content": ""}]
    prompt_with_trigger = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=tokenize)
    prompt_without_trigger = tokenizer.apply_chat_template(msg, add_generation_prompt=False, tokenize=tokenize)
    assistant_header = prompt_with_trigger[len(prompt_without_trigger):]
    return assistant_header


class Binarizer:
    def __init__(self):
        self.max_points = 1  # 0 and 1
        self.min_points = 0

    def fit(self, x):
        pass

    def transform(self, x):
        scalar = np.isscalar(x)
        if scalar:
            return 1 if x >= 0 else 0
        return (x >= 0).astype(int)


class LikertBinner:
    """Bins scores into 1...n_bins.

    - quantile_transform=False: round to nearest int, then clip to [1, n_bins].
    - quantile_transform=True: fit quantile-based cutpoints on training data (edges from quantiles over [min, max]).
    """
    def __init__(self, n_bins: int, quantile_transform: bool = False):
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2.")
        self.n_bins = self.max_points = n_bins  # points from 1 to n_bins
        self.min_points = 1
        self.quantile_transform = quantile_transform
        self._cutpoints = None  # internal cut points (len = n_bins-1)

    def fit(self, scores: np.ndarray):
        if self.quantile_transform:
            x = np.asarray(scores, dtype=float)
            if x.size == 0:
                raise ValueError("Empty input.")
            edges = np.quantile(x, np.linspace(0, 1, self.n_bins + 1))
            edges[0] = x.min()
            edges[-1] = x.max()
            self._cutpoints = edges[1:-1]  # n_bins-1 cutpoints

    def transform(self, scores: np.ndarray | float | List[float]):
        scalar = np.isscalar(scores)
        if scalar:
            arr = np.asarray([scores], dtype=float)
        else:
            arr = np.asarray(scores, dtype=float)

        if not self.quantile_transform:
            # round-half-away-from-zero
            rounded = np.where(arr >= 0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
            y = np.clip(rounded, 1, self.n_bins).astype(int)
        else:
            if self._cutpoints is None:
                raise ValueError("Call fit() before transform() when quantile_transform=True.")
            y = np.searchsorted(self._cutpoints, arr, side="right").astype(int) + 1
        return int(y[0]) if scalar else y


def find_tags_indices(tokens, tags):
    """
    Find occurrences of multiple tags in a single pass through the token sequence.

    Args:
        tokens (List[int]): A sequence of token IDs.
        tags (List[Tuple[str, List[int]]]): A list of tuples where each tuple is
            (tag_name, tag_token_sequence), e.g., ("[INST]", [733, 16289, 28793]).

    Returns:
        occurrences (Dict[str, List[Tuple[int, int]]]): A dictionary where each key is a tag name
            and the value is a list of tuples (start_index, end_index) where the tag was found.
    """
    occurrences = {tag[0]: [] for tag in tags}
    i = 0
    while i < len(tokens):
        match_found = False
        for tag_name, tag_tokens in tags:
            tag_len = len(tag_tokens)
            if tag_len == 0:
                continue
            # Check if there's enough tokens left and if they match the tag sequence.
            if i + tag_len <= len(tokens) and (tokens[i:i + tag_len] == tag_tokens).all():
                occurrences[tag_name].append((i, i + tag_len))
                # Move index past this tag and break to continue with the next token.
                i += tag_len
                match_found = True
                break
        if not match_found:
            i += 1
    return occurrences


def _make_labeler(score_scale: int, quantile_transform: bool):
    assert 2 <= score_scale < 10, "Score scale must be between 2 and 9."
    if score_scale == 2:
        return Binarizer()
    return LikertBinner(score_scale, quantile_transform)


def load_labeler(score_scale: int, quantile_transform: bool = True, save_dir: Path = None):
    labeler = _make_labeler(score_scale, quantile_transform)
    if score_scale == 2:  # binary
        exp_save_dir = save_dir
    else:
        exp_save_dir = save_dir.with_name(f'{save_dir.name}_{score_scale}points')
    os.makedirs(exp_save_dir, exist_ok=True)
    return labeler, exp_save_dir


def sample_layers(all_layers, layers: Union[int, List[int]] = 0) -> List[int]:
    if isinstance(layers, list):
        return layers
    elif layers == -1:  # only use the middle layer (one layer)
        selected_layers = [int(np.median(all_layers))]
    elif layers == 0:  # sample the layers (five layers)
        selected_layers = np.percentile(all_layers, [0, 25, 50, 75, 100], method='lower')
    else:
        selected_layers = [layers]
    return selected_layers


def format_pvalue(p):
    """Format p-value as p = 0.xx, p < 0.05, p < 0.01, p < 0.001, or p < 1e-N (down to a minimum of p < 1e-9)."""
    if p >= 0.05:
        return f'p = {p:.2f}'
    elif p >= 0.01:
        return 'p < 0.05'
    elif p >= 0.001:
        return 'p < 0.01'
    elif p >= 1e-4:
        return 'p < 0.001'
    elif p >= 1e-9:
        exp = int(-np.floor(np.log10(p)))
        return fr'$p < 10^{{-{exp}}}$'
    else:
        return r'$p < 10^{-9}$'
