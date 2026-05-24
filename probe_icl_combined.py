"""
Combined linear probe + ICL prediction experiment on LLM hidden states.

Outer/inner split logic mirrors probe_hidden.py.
For each data split:
  - Trains a logistic-regression probe on hidden states of the training set
  - Runs the LLM in ICL mode: training examples serve as labeled demonstrations,
    and each test example is predicted one-by-one (score token logit)
  - Saves a per-split cache containing: train/test indices (into the original df),
    probe classification report, model classification report, probe-vs-model agreement
  - Produces an aggregate results pkl and a printed summary table

The --layers flag restricts which label-layer score columns are evaluated
(default: all integer-typed columns in the pkl).

Usage:
    python probe_icl_combined.py \\
        --pkl_path results/.../hidden_mean_lr_example_scores.pkl \\
        --hidden_layer 16 \\
        --layers 5 10 15 20 \\
        [--n_outer 3] [--n_inner 5] \\
        [--max_icl_test 50] \\
        [--scenario ICL]

Note: each test example requires one model forward pass (after KV-caching the ICL
context), so keep --max_icl_test and the iteration counts modest for large models.
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neurofeedback import get_choice_scores, _init_neurofeedback_prompts
from probe_hidden import get_sequence_avg_hiddens, infer_model_name
from utils import load_lm, Binarizer, safe_dump

# ── defaults ──────────────────────────────────────────────────────────────────
N_POOL           = 500
N_TOTAL          = 600
TRAIN_SIZES      = [100, 200, 300, 400]
DEFAULT_OUTER    = 3
DEFAULT_INNER    = 5
DEFAULT_MAX_TEST = 50   # max test examples per ICL run (for tractability)


# ── ICL prediction on held-out test examples ──────────────────────────────────

@torch.inference_mode()
def predict_icl_on_test(
    model,
    tokenizer,
    labeler,
    train_sentences: list[str],
    train_labels: list[int],
    test_sentences: list[str],
    scenario: str = "ICL",
) -> list[int]:
    """
    Build an ICL context from (train_sentences, train_labels) and predict the
    score label for each sentence in test_sentences.

    The ICL context KV cache is computed once; each test example runs one forward
    pass over only the new tokens.

    Returns a list of integer predicted labels (same length as test_sentences).
    """
    prompt, meta_prompt = _init_neurofeedback_prompts(labeler, f"report_{scenario}")
    min_score = labeler.min_points
    max_score = labeler.max_points
    possible_choices = [str(i) for i in range(min_score, max_score + 1)]

    # Build ICL turns: alternating (user, assistant) message pairs
    icl_turns = []
    for sent, lbl in zip(train_sentences, train_labels):
        icl_turns.append({"role": "user",      "content": f"{prompt['user_msg']}\n"})
        icl_turns.append({"role": "assistant", "content": f"{sent} [Score: {lbl}]\n"})

    # Tokenise the full ICL context once
    icl_text = tokenizer.apply_chat_template(
        meta_prompt + icl_turns, tokenize=False, add_generation_prompt=False
    )
    icl_tokens = tokenizer(icl_text, return_tensors="pt").to(model.device)
    icl_len = icl_tokens["input_ids"].shape[1]

    # Prime the KV cache with the ICL context (one forward pass, all ICL tokens)
    icl_cache = DynamicCache()
    cache_position = torch.arange(0, icl_len, device=model.device)
    model(
        **icl_tokens,
        past_key_values=icl_cache,
        cache_position=cache_position,
        use_cache=True,
    )
    # icl_cache.get_seq_length() == icl_len from here on

    predicted_labels = []
    for test_sent in test_sentences:
        # Extend the prompt with the test sentence, stopping just after "Score: "
        # so the next predicted token is the score digit
        test_turns = [
            {"role": "user",      "content": f"{prompt['user_msg']}\n"},
            {"role": "assistant", "content": f"{test_sent} [Score: "},
        ]
        full_text = tokenizer.apply_chat_template(
            meta_prompt + icl_turns + test_turns,
            tokenize=False,
            continue_final_message=True,
        )
        full_tokens = tokenizer(full_text, return_tensors="pt").to(model.device)
        full_len = full_tokens["input_ids"].shape[1]

        # Only feed the NEW tokens; full attention mask covers cached + new tokens
        new_input_ids = full_tokens["input_ids"][:, icl_len:]
        cache_position = torch.arange(icl_len, full_len, device=model.device)

        outputs = model(
            input_ids=new_input_ids,
            attention_mask=full_tokens["attention_mask"],
            past_key_values=deepcopy(icl_cache),  # fresh copy per test example
            cache_position=cache_position,
            use_cache=True,
        )
        # The last position predicts the score token
        probs = get_choice_scores(
            outputs.logits[:, -1, :], tokenizer, possible_choices
        ).squeeze()
        predicted_labels.append(int(probs.argmax().item()) + min_score)

    return predicted_labels


# ── main experiment ───────────────────────────────────────────────────────────

def probe_icl_experiment(
    pkl_path: str | Path,
    hidden_layer: int,
    label_layers_filter: list[int] | None = None,
    n_outer: int = DEFAULT_OUTER,
    n_inner: int = DEFAULT_INNER,
    max_icl_test: int = DEFAULT_MAX_TEST,
    scenario: str = "ICL",
) -> list[dict]:
    pkl_path = Path(pkl_path)

    # 1. Load scores DataFrame ─────────────────────────────────────────────────
    df = load(pkl_path)
    assert "sentences" in df.columns, "Expected a 'sentences' column in the pkl"
    assert len(df) >= N_TOTAL, f"Expected at least {N_TOTAL} rows, got {len(df)}"

    all_label_layers = sorted(
        c for c in df.columns
        if c != "sentences" and np.issubdtype(type(c), np.integer)
    )
    if not all_label_layers:
        all_label_layers = sorted(c for c in df.columns if c != "sentences")

    if label_layers_filter is not None:
        label_layers = [ll for ll in label_layers_filter if ll in all_label_layers]
        missing = set(label_layers_filter) - set(label_layers)
        if missing:
            print(f"Warning: requested layers not found in pkl: {sorted(missing)}")
    else:
        label_layers = all_label_layers

    assert label_layers, "No label layers to evaluate"

    model_name = infer_model_name(pkl_path)
    sentences  = df["sentences"].tolist()
    n_total    = len(df)

    print(f"PKL           : {pkl_path}")
    print(f"Model         : {model_name}")
    print(f"Hidden layer  : {hidden_layer}")
    print(f"Label layers  : {label_layers}")
    print(f"Total rows    : {n_total}")
    print(f"n_outer/inner : {n_outer} / {n_inner}")
    print(f"max_icl_test  : {max_icl_test}")

    # 2. Hidden states for probe ───────────────────────────────────────────────
    model_safe = model_name.replace("/", "_")
    cache_path = pkl_path.parent / f"hiddens_{model_safe}_layer{hidden_layer}.npy"

    model = None
    tokenizer = None

    if cache_path.exists():
        print(f"\nLoading cached hiddens from {cache_path}")
        hiddens = np.load(cache_path)
    else:
        print("\nLoading model to extract hidden states ...")
        model, tokenizer = load_lm(model_name)
        model.eval()
        hiddens = get_sequence_avg_hiddens(model, tokenizer, sentences, layer=hidden_layer)
        np.save(cache_path, hiddens)
        print(f"Saved hiddens to {cache_path}")

    print(f"Hidden state shape: {hiddens.shape}")

    # 3. Load model for ICL (if not already loaded above) ─────────────────────
    if model is None:
        print("\nLoading model for ICL prediction ...")
        model, tokenizer = load_lm(model_name)
        model.eval()

    # 4. Pre-generate all random indices (deterministic) ──────────────────────
    rng = np.random.default_rng(42)
    outer_jobs = []
    for _ in range(n_outer):
        pool_idx = rng.choice(n_total, size=N_POOL, replace=False)
        inner_splits = [
            {s: rng.choice(N_POOL, size=s, replace=False) for s in TRAIN_SIZES}
            for _ in range(n_inner)
        ]
        outer_jobs.append((pool_idx, inner_splits))

    # 5. Per-split cache directory ─────────────────────────────────────────────
    save_dir = pkl_path.parent / f"probe_icl_splits_layer{hidden_layer}"
    save_dir.mkdir(exist_ok=True)

    records = []

    # 6. Main loops ────────────────────────────────────────────────────────────
    for outer_i, (pool_idx, inner_splits) in enumerate(outer_jobs):
        print(f"\n=== Outer {outer_i + 1}/{n_outer} ===")
        pool_hiddens   = hiddens[pool_idx]
        pool_sentences = [sentences[i] for i in pool_idx]

        for ll in label_layers:
            print(f"  Label layer {ll}")

            # Fit labeler on the pool sample for this label layer
            labeler    = Binarizer()
            pool_scores = df[ll].values[pool_idx]
            labeler.fit(pool_scores)
            pool_labels = labeler.transform(pool_scores)  # (N_POOL,) int array

            for inner_i, train_idx_by_size in enumerate(inner_splits):
                for train_size, train_idx in train_idx_by_size.items():

                    # Boolean mask over the pool for the test set
                    test_mask = np.ones(N_POOL, dtype=bool)
                    test_mask[train_idx] = False

                    # Original df indices
                    train_orig_idx = pool_idx[train_idx].tolist()
                    test_orig_idx  = pool_idx[test_mask].tolist()

                    split_id    = (f"outer{outer_i}_inner{inner_i}"
                                   f"_size{train_size}_layer{ll}")
                    split_cache = save_dir / f"{split_id}.pkl"

                    if split_cache.exists():
                        print(f"    [cached] {split_id}")
                        rec = load(split_cache)
                        records.append(rec)
                        continue

                    y_train = pool_labels[train_idx]
                    y_test  = pool_labels[test_mask]

                    if len(np.unique(y_train)) < 2:
                        print(f"    [skip – single class in train] {split_id}")
                        continue

                    # ── probe ─────────────────────────────────────────────────
                    scaler  = StandardScaler()
                    X_train = scaler.fit_transform(pool_hiddens[train_idx])
                    X_test  = scaler.transform(pool_hiddens[test_mask])

                    clf = SGDClassifier(max_iter=2000, n_jobs=1)
                    clf.fit(X_train, y_train)
                    probe_preds = clf.predict(X_test)
                    probe_report = classification_report(
                        y_test, probe_preds, output_dict=True, zero_division=0
                    )

                    # ── ICL prediction ─────────────────────────────────────────
                    train_sents  = [pool_sentences[i] for i in train_idx]
                    train_labels_icl = pool_labels[train_idx].tolist()

                    # All test sentences; optionally capped for speed
                    test_indices_pool = np.where(test_mask)[0]
                    if max_icl_test is not None and len(test_indices_pool) > max_icl_test:
                        rng_test = np.random.default_rng(
                            outer_i * 10000 + inner_i * 100 + train_size
                        )
                        test_indices_pool = rng_test.choice(
                            test_indices_pool, size=max_icl_test, replace=False
                        )
                        test_indices_pool = np.sort(test_indices_pool)

                    test_sents_icl = [pool_sentences[i] for i in test_indices_pool]
                    y_test_icl     = pool_labels[test_indices_pool]

                    print(
                        f"    ICL: outer={outer_i} inner={inner_i} "
                        f"size={train_size} layer={ll} "
                        f"(train={len(train_sents)}, test={len(test_sents_icl)})"
                    )
                    icl_preds = predict_icl_on_test(
                        model, tokenizer, labeler,
                        train_sents, train_labels_icl,
                        test_sents_icl, scenario,
                    )

                    model_report = classification_report(
                        y_test_icl, icl_preds, output_dict=True, zero_division=0
                    )

                    # Agreement: compare probe and ICL on the same test subset
                    probe_preds_icl = clf.predict(
                        scaler.transform(pool_hiddens[test_indices_pool])
                    )
                    try:
                        agreement = float(cohen_kappa_score(probe_preds_icl, icl_preds))
                    except ValueError:
                        # cohen_kappa undefined when one rater uses only one class
                        agreement = float("nan")
                        

                    rec = {
                        "outer_i":       outer_i,
                        "inner_i":       inner_i,
                        "train_size":    train_size,
                        "label_layer":   ll,
                        # indices into the ORIGINAL df
                        "train_idx":     train_orig_idx,
                        "test_idx":      test_orig_idx,
                        # subset used for ICL (may be capped)
                        "icl_test_idx":  pool_idx[test_indices_pool].tolist(),
                        # predictions & reports
                        "probe_preds":   probe_preds.tolist(),
                        "icl_preds":     icl_preds,
                        "true_labels_probe": y_test.tolist(),
                        "true_labels_icl":   y_test_icl.tolist(),
                        "probe_report":  probe_report,
                        "model_report":  model_report,
                        "kappa":         agreement,
                    }
                    safe_dump(rec, split_cache)
                    records.append(rec)

                    print(
                        f"    probe_acc={probe_report['accuracy']:.3f}  "
                        f"icl_acc={model_report['accuracy']:.3f}  "
                        f"kappa={agreement:.3f}"
                    )

    # 7. Aggregate and save ────────────────────────────────────────────────────
    agg_path = pkl_path.parent / f"probe_icl_results_hiddenlayer{hidden_layer}.pkl"
    safe_dump(records, agg_path)
    print(f"\nSaved {len(records)} split records to {agg_path}")

    # 8. Summary table ─────────────────────────────────────────────────────────
    if records:
        df_rec = pd.DataFrame([
            {
                "label_layer": r["label_layer"],
                "train_size":  r["train_size"],
                "probe_acc":   r["probe_report"]["accuracy"],
                "icl_acc":     r["model_report"]["accuracy"],
                "kappa":       r["kappa"],
            }
            for r in records
        ])
        print("\n=== Summary (mean ± SD) ===")
        col_w = 20
        print(
            f"  {'layer':>5}  {'N':>5}  "
            f"{'probe_acc':>{col_w}}  {'icl_acc':>{col_w}}  {'kappa':>{col_w}}"
        )
        for (ll, ts), grp in df_rec.groupby(["label_layer", "train_size"]):
            def fmt(col):
                return f"{grp[col].mean():.4f} ± {grp[col].std(ddof=1):.4f}"
            print(
                f"  {ll:>5}  {ts:>5}  "
                f"{fmt('probe_acc'):>{col_w}}  "
                f"{fmt('icl_acc'):>{col_w}}  "
                f"{fmt('kappa'):>{col_w}}"
            )

    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined probe + ICL prediction experiment per data split"
    )
    parser.add_argument(
        "--pkl_path", type=str, required=True,
        help="Path to hidden_mean_*_example_scores.pkl",
    )
    parser.add_argument(
        "--hidden_layer", type=int, required=True,
        help="Transformer layer index for hidden state extraction (1-based).",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Label-layer indices to evaluate (default: all integer columns in pkl).",
    )
    parser.add_argument(
        "--n_outer", type=int, default=DEFAULT_OUTER,
        help=f"Number of outer re-samples (pool of {N_POOL} from {N_TOTAL}). "
             f"Default: {DEFAULT_OUTER}",
    )
    parser.add_argument(
        "--n_inner", type=int, default=DEFAULT_INNER,
        help=f"Number of inner re-samples per train size. Default: {DEFAULT_INNER}",
    )
    parser.add_argument(
        "--max_icl_test", type=int, default=DEFAULT_MAX_TEST,
        help=(
            f"Max test examples per ICL run (for speed). "
            f"Set 0 for no limit. Default: {DEFAULT_MAX_TEST}"
        ),
    )
    parser.add_argument(
        "--scenario", type=str, default="ICL", choices=["ICL", "NF"],
        help="Prompt scenario for ICL. Default: ICL",
    )
    args = parser.parse_args()

    probe_icl_experiment(
        pkl_path=args.pkl_path,
        hidden_layer=args.hidden_layer,
        label_layers_filter=args.layers,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        max_icl_test=args.max_icl_test if args.max_icl_test > 0 else None,
        scenario=args.scenario,
    )
