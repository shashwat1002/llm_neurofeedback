"""
Probe-on-ICL experiment: train a linear probe to predict ICL model outputs
on held-out test hidden states.

Flow per split:
  1. Split pool into train (ICL context) and test sets.
  2. Run ICL on ALL test sentences → record ICL predicted labels.
  3. Split test set by --probe_test_ratio (default 0.5) into probe-train / probe-test.
  4. For each hidden layer in --layers:
     - Extract (or load cached) mean-pooled hidden states.
     - Oversample probe-train if classes are imbalanced.
     - Fit a linear SGDClassifier on probe-train hiddens → ICL labels.
     - Evaluate on probe-test hiddens; record accuracy.
  5. Repeat steps 2-4 for each label layer in --label_layers.
  6. Print mean ± std accuracy per (hidden_layer, train_size) cell,
     averaged across all label layers.

The probe is learning to *imitate* the ICL model's behaviour, not to predict
ground-truth scores directly.

Usage:
    python probe_on_icl_meta.py \\
        --pkl_path results/.../hidden_mean_lr_example_scores.pkl \\
        --label_layers 14 16 18 \\
        --layers 5 10 15 20 \\
        [--n_outer 3] [--n_inner 5] \\
        [--max_icl_test 100] \\
        [--probe_test_ratio 0.5] \\
        [--scenario ICL]
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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neurofeedback import get_choice_scores, _init_neurofeedback_prompts
from probe_hidden import get_sequence_avg_hiddens, infer_model_name
from utils import load_lm, Binarizer, safe_dump

# ── defaults ──────────────────────────────────────────────────────────────────
N_POOL                   = 5200
N_TOTAL                  = 6000
TRAIN_SIZES              = [100, 200, 500]
DEFAULT_OUTER            = 3
DEFAULT_INNER            = 5
DEFAULT_MAX_TEST         = 1000
DEFAULT_PROBE_TEST_RATIO = 0.2


# ── helpers ───────────────────────────────────────────────────────────────────

def probe_cosine_similarity(coef1: np.ndarray, coef2: np.ndarray) -> float:
    """
    Cosine similarity between two probe weight matrices (flattened).
    Returns NaN if shapes differ (different number of classes).
    """
    if coef1.shape != coef2.shape:
        return float("nan")
    w1 = coef1.flatten()
    w2 = coef2.flatten()
    denom = np.linalg.norm(w1) * np.linalg.norm(w2)
    if denom == 0:
        return 0.0
    return float(np.dot(w1, w2) / denom)


def oversample_to_balance(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample each minority class until all classes are equally sized."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() == counts.max():
        return X, y  # already balanced or single class

    max_count = counts.max()
    X_parts, y_parts = [], []
    for cls in classes:
        mask = y == cls
        X_cls, y_cls = X[mask], y[mask]
        if len(y_cls) < max_count:
            X_cls, y_cls = resample(
                X_cls, y_cls,
                n_samples=max_count,
                replace=True,
                random_state=random_state,
            )
        X_parts.append(X_cls)
        y_parts.append(y_cls)
    return np.vstack(X_parts), np.concatenate(y_parts)


# ── ICL prediction (reused from probe_icl_combined) ──────────────────────────

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

    The ICL context KV cache is computed once; each test example runs one
    forward pass over only the new tokens.
    """
    prompt, meta_prompt = _init_neurofeedback_prompts(labeler, f"report_{scenario}")
    min_score = labeler.min_points
    max_score = labeler.max_points
    possible_choices = [str(i) for i in range(min_score, max_score + 1)]

    icl_turns = []
    for sent, lbl in zip(train_sentences, train_labels):
        icl_turns.append({"role": "user",      "content": f"{prompt['user_msg']}\n"})
        icl_turns.append({"role": "assistant", "content": f"{sent} [Score: {lbl}]\n"})

    icl_text = tokenizer.apply_chat_template(
        meta_prompt + icl_turns, tokenize=False, add_generation_prompt=False
    )
    icl_tokens = tokenizer(icl_text, return_tensors="pt").to(model.device)
    icl_len = icl_tokens["input_ids"].shape[1]

    icl_cache = DynamicCache()
    cache_position = torch.arange(0, icl_len, device=model.device)
    model(
        **icl_tokens,
        past_key_values=icl_cache,
        cache_position=cache_position,
        use_cache=True,
    )

    predicted_labels = []
    for test_sent in test_sentences:
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

        new_input_ids = full_tokens["input_ids"][:, icl_len:]
        cache_position = torch.arange(icl_len, full_len, device=model.device)

        outputs = model(
            input_ids=new_input_ids,
            attention_mask=full_tokens["attention_mask"],
            past_key_values=deepcopy(icl_cache),
            cache_position=cache_position,
            use_cache=True,
        )
        probs = get_choice_scores(
            outputs.logits[:, -1, :], tokenizer, possible_choices
        ).squeeze()
        predicted_labels.append(int(probs.argmax().item()) + min_score)

    return predicted_labels


# ── main experiment ───────────────────────────────────────────────────────────

def probe_on_icl_experiment(
    pkl_path: str | Path,
    label_layers: list[int],
    hidden_layers: list[int],
    n_outer: int = DEFAULT_OUTER,
    n_inner: int = DEFAULT_INNER,
    max_icl_test: int | None = DEFAULT_MAX_TEST,
    probe_test_ratio: float = DEFAULT_PROBE_TEST_RATIO,
    scenario: str = "ICL",
) -> list[dict]:
    pkl_path = Path(pkl_path)

    # 1. Load scores DataFrame ─────────────────────────────────────────────────
    df = load(pkl_path)
    assert "sentences" in df.columns, "Expected a 'sentences' column in the pkl"
    assert len(df) >= N_TOTAL, f"Expected at least {N_TOTAL} rows, got {len(df)}"
    for ll in label_layers:
        assert ll in df.columns, (
            f"label_layer={ll} not found in pkl columns: {list(df.columns)}"
        )

    model_name = infer_model_name(pkl_path)
    sentences  = df["sentences"].tolist()
    n_total    = len(df)

    print(f"PKL              : {pkl_path}")
    print(f"Model            : {model_name}")
    print(f"Label layers     : {label_layers}")
    print(f"Hidden layers    : {hidden_layers}")
    print(f"Total rows       : {n_total}")
    print(f"n_outer/inner    : {n_outer} / {n_inner}")
    print(f"max_icl_test     : {max_icl_test}")
    print(f"probe_test_ratio : {probe_test_ratio}")

    # 2. Load / extract hidden states for each requested layer ─────────────────
    model_safe = model_name.replace("/", "_")
    all_hiddens: dict[int, np.ndarray] = {}

    model     = None
    tokenizer = None

    for hl in hidden_layers:
        cache_path = pkl_path.parent / f"hiddens_{model_safe}_layer{hl}.npy"
        if cache_path.exists():
            print(f"\nLoading cached hiddens (layer {hl}) from {cache_path}")
            all_hiddens[hl] = np.load(cache_path)
        else:
            if model is None:
                print("\nLoading model to extract hidden states …")
                model, tokenizer = load_lm(model_name)
                model.eval()
            print(f"Extracting hidden states for layer {hl} …")
            hiddens = get_sequence_avg_hiddens(
                model, tokenizer, sentences, layer=hl
            )
            np.save(cache_path, hiddens)
            print(f"Saved hiddens to {cache_path}")
            all_hiddens[hl] = hiddens

    for hl, h in all_hiddens.items():
        print(f"Hidden state shape (layer {hl}): {h.shape}")

    # 3. Load model for ICL (if not already loaded above) ─────────────────────
    if model is None:
        print("\nLoading model for ICL prediction …")
        model, tokenizer = load_lm(model_name)
        model.eval()

    # 4. Pre-generate all random indices (deterministic, shared across label layers)
    rng = np.random.default_rng(42)
    outer_jobs = []
    for _ in range(n_outer):
        pool_idx = rng.choice(n_total, size=N_POOL, replace=False)
        inner_splits = [
            {s: rng.choice(N_POOL, size=s, replace=False) for s in TRAIN_SIZES}
            for _ in range(n_inner)
        ]
        outer_jobs.append((pool_idx, inner_splits))

    all_records = []

    # 5. Loop over each label layer ────────────────────────────────────────────
    for label_layer in label_layers:
        print(f"\n{'='*60}")
        print(f"=== Label layer {label_layer} ===")
        print(f"{'='*60}")

        # Cache directory per label layer
        save_dir = pkl_path.parent / f"probe_on_icl_splits_labellayer{label_layer}"
        save_dir.mkdir(exist_ok=True)

        records = []

        # 6. Main loops ────────────────────────────────────────────────────────
        for outer_i, (pool_idx, inner_splits) in enumerate(outer_jobs):
            print(f"\n=== Outer {outer_i + 1}/{n_outer} (label_layer={label_layer}) ===")
            pool_sentences = [sentences[i] for i in pool_idx]

            # Fit labeler on pool scores for this label layer
            labeler     = Binarizer()
            pool_scores = df[label_layer].values[pool_idx]
            labeler.fit(pool_scores)
            pool_labels = labeler.transform(pool_scores)  # (N_POOL,) int array

            for inner_i, train_idx_by_size in enumerate(inner_splits):
                for train_size, train_idx in train_idx_by_size.items():

                    test_mask         = np.ones(N_POOL, dtype=bool)
                    test_mask[train_idx] = False
                    test_indices_pool = np.where(test_mask)[0]

                    # Optionally cap the number of ICL test samples for speed
                    if max_icl_test is not None and len(test_indices_pool) > max_icl_test:
                        rng_test = np.random.default_rng(
                            outer_i * 10000 + inner_i * 100 + train_size
                        )
                        test_indices_pool = rng_test.choice(
                            test_indices_pool, size=max_icl_test, replace=False
                        )
                        test_indices_pool = np.sort(test_indices_pool)

                    n_test = len(test_indices_pool)

                    # ── Run ICL on the test samples ────────────────────────
                    icl_cache_id   = (f"outer{outer_i}_inner{inner_i}"
                                      f"_size{train_size}_labellayer{label_layer}")
                    icl_cache_file = save_dir / f"icl_{icl_cache_id}.pkl"

                    if icl_cache_file.exists():
                        print(f"  [icl cached] {icl_cache_id}")
                        icl_preds = load(icl_cache_file)
                    else:
                        train_sents      = [pool_sentences[i] for i in train_idx]
                        train_labels_icl = pool_labels[train_idx].tolist()
                        test_sents_icl   = [pool_sentences[i] for i in test_indices_pool]

                        print(
                            f"  ICL: outer={outer_i} inner={inner_i} "
                            f"size={train_size} labellayer={label_layer} "
                            f"(train={len(train_sents)}, test={n_test})"
                        )
                        icl_preds = predict_icl_on_test(
                            model, tokenizer, labeler,
                            train_sents, train_labels_icl,
                            test_sents_icl, scenario,
                        )
                        safe_dump(icl_preds, icl_cache_file)

                    icl_preds = np.array(icl_preds)
                    n_test = len(icl_preds)  # reconcile with cached length

                    # ICL accuracy vs ground-truth labels
                    icl_acc = float(accuracy_score(pool_labels[test_indices_pool], icl_preds))

                    # ── Split test set for probe train / test ──────────────
                    n_probe_train = max(1, int(np.floor(n_test * (1 - probe_test_ratio))))
                    rng_probe = np.random.default_rng(
                        outer_i * 100000 + inner_i * 1000 + train_size
                    )
                    perm = rng_probe.permutation(n_test)
                    probe_train_sel = perm[:n_probe_train]   # indices into test_indices_pool
                    probe_test_sel  = perm[n_probe_train:]

                    if len(probe_test_sel) == 0:
                        print(f"  [skip – not enough test samples] {icl_cache_id}")
                        continue

                    y_probe_train = icl_preds[probe_train_sel]
                    y_probe_test  = icl_preds[probe_test_sel]

                    if len(np.unique(y_probe_train)) < 2:
                        print(f"  [skip – single ICL class in probe-train] {icl_cache_id}")
                        continue

                    # Pool indices for the probe subsets
                    probe_train_pool_idx = test_indices_pool[probe_train_sel]
                    probe_test_pool_idx  = test_indices_pool[probe_test_sel]

                    # Ground-truth cluster labels for the probe subsets
                    y_cluster_train = pool_labels[probe_train_pool_idx]
                    y_cluster_test  = pool_labels[probe_test_pool_idx]

                    # ── Per-layer probe ────────────────────────────────────
                    for hl in hidden_layers:
                        split_id    = (f"outer{outer_i}_inner{inner_i}"
                                       f"_size{train_size}_labellayer{label_layer}"
                                       f"_hiddenlayer{hl}")
                        # v2: includes cluster probe + cosine similarity
                        split_cache = save_dir / f"probe_{split_id}_v2.pkl"

                        if split_cache.exists():
                            print(f"  [probe cached] {split_id}")
                            rec = load(split_cache)
                            records.append(rec)
                            continue

                        hiddens = all_hiddens[hl]

                        scaler        = StandardScaler()
                        X_probe_train = scaler.fit_transform(
                            hiddens[pool_idx[probe_train_pool_idx]]
                        )
                        X_probe_test  = scaler.transform(
                            hiddens[pool_idx[probe_test_pool_idx]]
                        )

                        # ── Meta probe: predicts ICL outputs ──────────────
                        # X_bal_meta, y_bal_meta = oversample_to_balance(
                        #     X_probe_train, y_probe_train,
                        #     random_state=outer_i * 100 + inner_i,
                        # )
                        clf_meta = SGDClassifier(
                            loss="log_loss",       # or "hinge" for linear SVM
                            penalty="l2",
                            alpha=1e-2,
                            learning_rate="optimal",
                            max_iter=1000,
                            tol=1e-3,
                            early_stopping=True,
                            n_iter_no_change=5,
                            random_state=42,
                        )
                        print("fitting... (meta probe to predict ICL labels)")
                        print(X_probe_train.shape, y_probe_train.shape)
                        print(X_probe_test.shape, y_probe_test.shape)

                        clf_meta.fit(X_probe_train, y_probe_train)

                        probe_preds = clf_meta.predict(X_probe_test)
                        acc = float(accuracy_score(y_probe_test, probe_preds))
                        print("train acc:", clf_meta.score(X_probe_train, y_probe_train))

                        # ── Cluster probe: predicts ground-truth labels ────
                        cluster_acc            = float("nan")
                        cosine_sim             = float("nan")
                        cluster_icl_agreement  = float("nan")
                        if len(np.unique(y_cluster_train)) >= 2:

                            clf_cluster = SGDClassifier(
                            loss="log_loss",       # or "hinge" for linear SVM
                            penalty="l2",
                            alpha=1e-3,
                            learning_rate="optimal",
                            max_iter=1000,
                            tol=1e-5,
                            early_stopping=True,
                            n_iter_no_change=5,
                            random_state=42,
                            )
                            clf_cluster.fit(X_probe_train, y_cluster_train)

                            cluster_preds         = clf_cluster.predict(X_probe_test)
                            cluster_acc           = float(accuracy_score(y_cluster_test, cluster_preds))
                            cluster_icl_agreement = float(accuracy_score(y_probe_test, cluster_preds))
                            cosine_sim            = probe_cosine_similarity(
                                clf_meta.coef_, clf_cluster.coef_
                            )

                        rec = {
                            "outer_i":              outer_i,
                            "inner_i":              inner_i,
                            "train_size":           train_size,
                            "label_layer":          label_layer,
                            "hidden_layer":         hl,
                            # original df indices
                            "icl_test_orig_idx":    pool_idx[test_indices_pool].tolist(),
                            "probe_train_orig_idx": pool_idx[probe_train_pool_idx].tolist(),
                            "probe_test_orig_idx":  pool_idx[probe_test_pool_idx].tolist(),
                            # labels: ICL outputs (not ground truth)
                            "icl_preds":            icl_preds.tolist(),
                            "y_probe_train":        y_probe_train.tolist(),
                            "y_probe_test":         y_probe_test.tolist(),
                            "probe_preds":          probe_preds.tolist(),
                            "accuracy":             acc,
                            "icl_accuracy":         icl_acc,
                            # cluster probe
                            "y_cluster_train":      y_cluster_train.tolist(),
                            "y_cluster_test":       y_cluster_test.tolist(),
                            "cluster_acc":           cluster_acc,
                            "cluster_icl_agreement": cluster_icl_agreement,
                            "probe_cosine_sim":      cosine_sim,
                        }
                        safe_dump(rec, split_cache)
                        records.append(rec)

                        print(
                            f"  icl_acc={icl_acc:.3f}  "
                            f"meta_probe_acc={acc:.3f}  "
                            f"cluster_probe_acc={cluster_acc:.3f}  "
                            f"cluster_icl_agree={cluster_icl_agreement:.3f}  "
                            f"cosine_sim={cosine_sim:.3f}  "
                            f"(label_layer={label_layer}, hidden_layer={hl}, "
                            f"train_size={train_size})"
                        )

        # Save per-label-layer records
        agg_path = pkl_path.parent / f"probe_on_icl_results_labellayer{label_layer}.pkl"
        safe_dump(records, agg_path)
        print(f"\nSaved {len(records)} split records for label_layer={label_layer} to {agg_path}")
        all_records.extend(records)

    # 7. Aggregate all records and save ───────────────────────────────────────
    layers_str = "_".join(str(ll) for ll in sorted(label_layers))
    agg_all_path = pkl_path.parent / f"probe_on_icl_results_labellayers{layers_str}.pkl"
    safe_dump(all_records, agg_all_path)
    print(f"\nSaved {len(all_records)} total records to {agg_all_path}")

    # 8. Summary table (averaged across label layers) ─────────────────────────
    if all_records:
        df_rec = pd.DataFrame([
            {
                "hidden_layer":  r["hidden_layer"],
                "train_size":    r["train_size"],
                "label_layer":   r["label_layer"],
                "accuracy":      r["accuracy"],
                "icl_accuracy":  r.get("icl_accuracy",      float("nan")),
                "cluster_acc":           r.get("cluster_acc",            float("nan")),
                "cluster_icl_agreement": r.get("cluster_icl_agreement",  float("nan")),
                "cosine_sim":            r.get("probe_cosine_sim",        float("nan")),
            }
            for r in all_records
        ])

        col_w = 22
        n_ll  = len(label_layers)
        print(f"\n=== Summary averaged across {n_ll} label layer(s): {label_layers} ===")
        print(f"  {'hidden_layer':>12}  {'train_size':>10}  "
              f"{'icl_acc':>{col_w}}  {'meta_probe_acc':>{col_w}}  "
              f"{'cluster_probe_acc':>{col_w}}  {'cluster_icl_agree':>{col_w}}  "
              f"{'cosine_sim':>{col_w}}")
        for (hl, ts), grp in df_rec.groupby(["hidden_layer", "train_size"]):
            def _fmt(col):
                vals = grp[col].dropna()
                if len(vals) == 0:
                    return "      n/a      "
                m, s = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
                return f"{m:.4f} ± {s:.4f}"
            print(
                f"  {hl:>12}  {ts:>10}  "
                f"{_fmt('icl_accuracy'):>{col_w}}  "
                f"{_fmt('accuracy'):>{col_w}}  "
                f"{_fmt('cluster_acc'):>{col_w}}  "
                f"{_fmt('cluster_icl_agreement'):>{col_w}}  "
                f"{_fmt('cosine_sim'):>{col_w}}"
            )

    return all_records


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train a linear probe to predict ICL model outputs "
            "from hidden states on held-out test examples."
        )
    )
    parser.add_argument(
        "--pkl_path", type=str, required=True,
        help="Path to hidden_mean_*_example_scores.pkl",
    )
    parser.add_argument(
        "--label_layers", type=int, nargs="+", required=True,
        help=(
            "Score column(s) (label layers) in the pkl used to binarize labels for ICL. "
            "Results are averaged across all specified layers."
        ),
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True,
        help="Transformer layer indices to extract hidden states from (1-based).",
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
        "--probe_test_ratio", type=float, default=DEFAULT_PROBE_TEST_RATIO,
        help=(
            "Fraction of ICL test samples held out for probe evaluation. "
            f"Default: {DEFAULT_PROBE_TEST_RATIO}"
        ),
    )
    parser.add_argument(
        "--scenario", type=str, default="ICL", choices=["ICL", "NF"],
        help="Prompt scenario for ICL. Default: ICL",
    )
    args = parser.parse_args()

    probe_on_icl_experiment(
        pkl_path=args.pkl_path,
        label_layers=args.label_layers,
        hidden_layers=args.layers,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        max_icl_test=args.max_icl_test if args.max_icl_test > 0 else None,
        probe_test_ratio=args.probe_test_ratio,
        scenario=args.scenario,
    )
