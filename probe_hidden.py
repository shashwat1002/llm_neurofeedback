"""
Linear probe experiment on LLM hidden states.

Given a hidden_mean_{CLF}_example_scores.pkl file:
  - Infers model name from the folder structure
  - Extracts last-token hidden states at a single transformer layer (--layer)
  - Loops over ALL score columns in the pkl as candidate label layers
  - Outer loop: re-samples 500 examples from the full 600
  - Inner loop: re-samples training splits of sizes [100, 200, 300, 400]
  - Trains a logistic-regression probe per label layer and evaluates on held-out examples
  - Reports mean ± SD accuracy per label layer × training set size

Usage:
    python probe_hidden.py --pkl_path <pkl_path> --layer <L> [--n_outer 5] [--n_inner 10]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from joblib import load, Parallel, delayed
from sklearn.linear_model import SGDClassifier


from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MODEL_ALIASES, load_lm, Binarizer

# ── defaults ──────────────────────────────────────────────────────────────────
N_POOL        = 500
N_TOTAL       = 600
TRAIN_SIZES   = [100, 200, 300, 400]
DEFAULT_OUTER = 5
DEFAULT_INNER = 10
BATCH_SIZE    = 16


# ── helpers ───────────────────────────────────────────────────────────────────

def infer_model_name(pkl_path: Path) -> str:
    """
    Infer the HuggingFace model id from the folder containing the pkl.
    Expected structure:
        results/{model_folder}/{dataset}/hidden_mean_*_example_scores.pkl
    where model_folder is the HF id with '/' replaced by '_'.
    """
    model_folder = pkl_path.parents[1].name
    for hf_name in MODEL_ALIASES.values():
        if hf_name.replace("/", "_") == model_folder:
            return hf_name
    return model_folder.replace("_", "/", 1)


@torch.inference_mode()
def get_last_token_hiddens(
    model,
    tokenizer,
    sentences: list[str],
    layer: int,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Return the last-token hidden state at transformer layer `layer` for every sentence.

    `layer` is 1-based (1 = first transformer block; model.config.num_hidden_layers = last).
    Returns array of shape (N, hidden_dim).
    """
    all_hiddens = []
    n_batches = (len(sentences) + batch_size - 1) // batch_size

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[layer]                         # (batch, seq, hidden)
        last_idx = inputs["attention_mask"].sum(dim=1) - 1        # (batch,)
        last_hs  = hs[torch.arange(hs.size(0)), last_idx, :]     # (batch, hidden)
        all_hiddens.append(last_hs.cpu().float())

        batch_num = i // batch_size + 1
        if batch_num % 10 == 0:
            print(f"  batch {batch_num}/{n_batches}")

    return torch.cat(all_hiddens, dim=0).numpy()   # (N, hidden_dim)


@torch.inference_mode()
def get_sequence_avg_hiddens(
    model,
    tokenizer,
    sentences: list[str],
    layer: int,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Return the mean-pooled hidden state at transformer layer `layer` for every sentence.

    Averages over all non-padding token positions (as indicated by the attention mask).
    `layer` is 1-based (1 = first transformer block; model.config.num_hidden_layers = last).
    Returns array of shape (N, hidden_dim).
    """
    all_hiddens = []
    n_batches = (len(sentences) + batch_size - 1) // batch_size

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        hs   = outputs.hidden_states[layer]              # (batch, seq, hidden)
        mask = inputs["attention_mask"].unsqueeze(-1)    # (batch, seq, 1)
        avg_hs = (hs * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, hidden)
        all_hiddens.append(avg_hs.cpu().float())

        batch_num = i // batch_size + 1
        if batch_num % 10 == 0:
            print(f"  batch {batch_num}/{n_batches}")

    return torch.cat(all_hiddens, dim=0).numpy()   # (N, hidden_dim)


# ── parallelisable worker ──────────────────────────────────────────────────────

def _run_outer(
    pool_idx: np.ndarray,
    train_idx_by_inner: list[dict[int, np.ndarray]],
    hiddens: np.ndarray,
    scores_by_layer: dict[int, np.ndarray],
    label_layers: list,
) -> dict[int, dict[int, list[float]]]:
    """Process one outer re-sample: all inner loops and train sizes for every label layer."""
    pool_hiddens = hiddens[pool_idx]
    results: dict[int, dict[int, list[float]]] = {
        ll: {s: [] for s in TRAIN_SIZES} for ll in label_layers
    }

    # Fit labeller on the pool sample, then binarize pool labels
    labeler = Binarizer()
    pool_scores_by_layer = {ll: scores_by_layer[ll][pool_idx] for ll in label_layers}
    for ll in label_layers:
        labeler.fit(pool_scores_by_layer[ll])
    labels_by_layer = {ll: labeler.transform(pool_scores_by_layer[ll]) for ll in label_layers}

    for train_idx_by_size in train_idx_by_inner:
        for train_size, train_idx in train_idx_by_size.items():
            test_mask = np.ones(len(pool_idx), dtype=bool)
            test_mask[train_idx] = False

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(pool_hiddens[train_idx])
            X_test  = scaler.transform(pool_hiddens[test_mask])

            for ll in label_layers:
                y_train = labels_by_layer[ll][train_idx]
                y_test  = labels_by_layer[ll][test_mask]

                if len(np.unique(y_train)) < 2:
                    continue

                clf = SGDClassifier(max_iter=2000, n_jobs=1)
                clf.fit(X_train, y_train)
                results[ll][train_size].append(clf.score(X_test, y_test))
    print(results)
    return results


# ── main experiment ───────────────────────────────────────────────────────────

def probe_experiment(
    pkl_path: str | Path,
    hidden_layer: int,
    n_outer: int = DEFAULT_OUTER,
    n_inner: int = DEFAULT_INNER,
    n_jobs: int = -1,
) -> dict[int, dict[int, list[float]]]:
    """
    Returns nested dict: results[label_layer][train_size] = list of accuracy values.
    """
    pkl_path = Path(pkl_path)

    # 1. Load scores DataFrame
    df = load(pkl_path)
    assert "sentences" in df.columns, "Expected a 'sentences' column in the pkl DataFrame"
    assert len(df) >= N_TOTAL, f"Expected at least {N_TOTAL} rows, got {len(df)}"

    # Identify all numeric score columns (i.e. label layer candidates)
    label_layers = sorted(c for c in df.columns if c != "sentences" and np.issubdtype(type(c), np.integer))
    if not label_layers:
        # fallback: any non-sentences column that is numeric dtype
        label_layers = sorted(c for c in df.columns if c != "sentences")
    assert label_layers, "No score columns found in the DataFrame"

    # 2. Log setup
    model_name = infer_model_name(pkl_path)
    print(f"PKL file      : {pkl_path}")
    print(f"Model name    : {model_name}")
    print(f"Hidden layer  : {hidden_layer}")
    print(f"Label layers  : {label_layers}")
    print(f"Total rows    : {len(df)}")

    sentences = df["sentences"].tolist()

    # 3. Load or compute hidden states for the fixed layer
    model_safe = model_name.replace("/", "_")
    cache_path = pkl_path.parent / f"hiddens_{model_safe}_layer{hidden_layer}.npy"

    if cache_path.exists():
        print(f"\nLoading cached hidden states from {cache_path} …")
        hiddens = np.load(cache_path)
        print(f"Hidden state array shape: {hiddens.shape}")
    else:
        print("\nLoading model …")
        model, tokenizer = load_lm(model_name)
        model.eval()
        print(f"Extracting hidden states at layer {hidden_layer} …")
        print("using avg")
        hiddens = get_sequence_avg_hiddens(model, tokenizer, sentences, layer=hidden_layer)
        print(f"Hidden state array shape: {hiddens.shape}")
        np.save(cache_path, hiddens)
        print(f"Cached hidden states to {cache_path}")

    # 4. Store raw scores; labeller will be fit per outer iteration on the pool sample
    scores_by_layer: dict[int, np.ndarray] = {
        ll: df[ll].values for ll in label_layers
    }

    # Pre-generate all random indices so the RNG sequence is deterministic
    rng = np.random.default_rng(42)
    n_total = len(df)

    outer_jobs = []
    for _ in range(n_outer):
        pool_idx = rng.choice(n_total, size=N_POOL, replace=False)
        train_idx_by_inner = [
            {s: rng.choice(N_POOL, size=s, replace=False) for s in TRAIN_SIZES}
            for _ in range(n_inner)
        ]
        outer_jobs.append((pool_idx, train_idx_by_inner))

    # 5. Parallel outer loop
    print(f"\nRunning {n_outer} outer iterations in parallel (n_jobs={n_jobs}) …")
    outer_results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_run_outer)(pool_idx, train_idx_by_inner, hiddens, scores_by_layer, label_layers)
        for pool_idx, train_idx_by_inner in outer_jobs
    )

    # Merge results from all outer iterations
    results: dict[int, dict[int, list[float]]] = {
        ll: {s: [] for s in TRAIN_SIZES} for ll in label_layers
    }
    for res in outer_results:
        for ll in label_layers:
            for s in TRAIN_SIZES:
                results[ll][s].extend(res[ll][s])

    # 6. Summary table: rows = label_layers, cols = train sizes
    col_w = 18
    print(f"\n=== Results (mean ± SD accuracy, hidden_layer={hidden_layer}) ===")
    header = [f"{'LabelLayer':>12}"] + [f"{'N='+str(s):>{col_w}}" for s in TRAIN_SIZES]
    print("  ".join(header))
    print("-" * (14 + (col_w + 2) * len(TRAIN_SIZES)))

    for ll in label_layers:
        row = [f"{ll:>12}"]
        for train_size in TRAIN_SIZES:
            accs = np.array(results[ll][train_size])
            cell = f"{accs.mean():.4f} ± {accs.std(ddof=1):.4f}" if len(accs) else "N/A"
            row.append(f"{cell:>{col_w}}")
        print("  ".join(row))

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear probe on LLM hidden states: fixed layer, all label layers"
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        required=True,
        help="Path to hidden_mean_{CLF}_example_scores.pkl",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help=(
            "Transformer layer index for hidden state extraction. "
            "1 = first block, model.config.num_hidden_layers = last block."
        ),
    )
    parser.add_argument(
        "--n_outer",
        type=int,
        default=DEFAULT_OUTER,
        help=f"Number of outer re-samples (pool of {N_POOL} from {N_TOTAL}). Default: {DEFAULT_OUTER}",
    )
    parser.add_argument(
        "--n_inner",
        type=int,
        default=DEFAULT_INNER,
        help=f"Number of inner re-samples per training size. Default: {DEFAULT_INNER}",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for outer loop (-1 = all cores). Default: -1",
    )
    args = parser.parse_args()

    results = probe_experiment(
        pkl_path=args.pkl_path,
        hidden_layer=args.layer,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        n_jobs=args.n_jobs,
    )
    import pandas as pd
    df_results = pd.DataFrame([
        {"label_layer": ll, "train_size": s, "accuracy": acc}
        for ll, size_dict in results.items()
        for s, accs in size_dict.items()
        for acc in accs])
    df_results.to_csv(Path(args.pkl_path).with_suffix(".probe_results.csv"), index=False)
