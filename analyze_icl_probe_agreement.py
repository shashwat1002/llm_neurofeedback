"""
Population agreement analysis between meta-probe and ICL predictions.

For each split's probe-test samples, partitions them into:
  - ICL-correct: samples where ICL prediction == ground truth
  - ICL-wrong:   samples where ICL prediction != ground truth

Then reports meta-probe accuracy (vs ICL label) and vs ground truth
in each partition, aggregated over all cached v2 split files.

Usage (single layer):
    python analyze_icl_probe_agreement.py \\
        --split_dir results/.../probe_on_icl_splits_labellayer4 \\
        [--hidden_layer 0] \\
        [--label_layer 4]

Usage (multiple layers — per-layer + cross-layer aggregate):
    python analyze_icl_probe_agreement.py \\
        --base_dir results/.../ \\
        --label_layers 4 8 16 25 \\
        [--hidden_layer 0]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load


def load_split_records(
    split_dir: Path, hidden_layer: int, label_layer: int
) -> list[dict]:
    pattern = f"probe_outer*_inner*_size*_labellayer{label_layer}_hiddenlayer{hidden_layer}_v2.pkl"
    files = sorted(split_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No v2 probe cache files found in {split_dir} "
            f"matching hidden_layer={hidden_layer}, label_layer={label_layer}"
        )
    print(f"Found {len(files)} cached split files.")
    return [load(f) for f in files]


def analyse_record(rec: dict) -> dict | None:
    probe_preds    = np.array(rec["probe_preds"])      # meta probe's output
    y_icl          = np.array(rec["y_probe_test"])     # ICL prediction (probe target)
    y_gt           = np.array(rec["y_cluster_test"])   # ground truth label

    n = len(y_icl)
    if n == 0:
        return None

    icl_correct_mask = y_icl == y_gt
    icl_wrong_mask   = ~icl_correct_mask

    def _acc(pred, target, mask):
        if mask.sum() == 0:
            return float("nan")
        return float((pred[mask] == target[mask]).mean())

    all_mask = np.ones(n, dtype=bool)
    return {
        "outer_i":    rec["outer_i"],
        "inner_i":    rec["inner_i"],
        "train_size": rec["train_size"],
        "n_probe_test":         n,
        "n_icl_correct":        int(icl_correct_mask.sum()),
        "n_icl_wrong":          int(icl_wrong_mask.sum()),
        # ICL accuracy on this probe-test subset
        "icl_acc":              float(icl_correct_mask.mean()),
        # --- Overall (no partition) ---
        "probe_acc_vs_icl":     _acc(probe_preds, y_icl, all_mask),
        "probe_acc_vs_gt":      _acc(probe_preds, y_gt,  all_mask),
        # --- ICL-correct subpopulation ---
        "probe_acc_icl_correct_vs_icl": _acc(probe_preds, y_icl, icl_correct_mask),
        "probe_acc_icl_correct_vs_gt":  _acc(probe_preds, y_gt,  icl_correct_mask),
        # --- ICL-wrong subpopulation ---
        "probe_acc_icl_wrong_vs_icl":   _acc(probe_preds, y_icl, icl_wrong_mask),
        "probe_acc_icl_wrong_vs_gt":    _acc(probe_preds, y_gt,  icl_wrong_mask),
    }


def print_summary(df: pd.DataFrame) -> None:
    col_w = 22

    def _fmt(series):
        v = series.dropna()
        if len(v) == 0:
            return "       n/a       "
        m, s = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
        return f"{m:.4f} ± {s:.4f}"

    # ── Overall (no partition) ────────────────────────────────────────────────
    print("\n=== OVERALL (all probe-test samples) ===")
    print(f"  {'train_size':>10}  {'avg n':>7}  "
          f"{'probe vs ICL':>{col_w}}  {'probe vs GT':>{col_w}}")
    print("  " + "-" * (10 + 7 + col_w * 2 + 10))
    for ts, grp in df.groupby("train_size"):
        print(
            f"  {ts:>10}  {grp['n_probe_test'].mean():>7.1f}  "
            f"{_fmt(grp['probe_acc_vs_icl']):>{col_w}}  "
            f"{_fmt(grp['probe_acc_vs_gt']):>{col_w}}"
        )
    print(
        f"  {'(pooled)':>10}  {df['n_probe_test'].mean():>7.1f}  "
        f"{_fmt(df['probe_acc_vs_icl']):>{col_w}}  "
        f"{_fmt(df['probe_acc_vs_gt']):>{col_w}}"
    )

    for label, n_col, vs_icl_col, vs_gt_col in [
        ("ICL CORRECT (ICL pred == GT)", "n_icl_correct",
         "probe_acc_icl_correct_vs_icl", "probe_acc_icl_correct_vs_gt"),
        ("ICL WRONG   (ICL pred != GT)", "n_icl_wrong",
         "probe_acc_icl_wrong_vs_icl",   "probe_acc_icl_wrong_vs_gt"),
    ]:
        print(f"\n=== {label} ===")
        print(f"  {'train_size':>10}  {'avg n':>7}  "
              f"{'probe vs ICL':>{col_w}}  {'probe vs GT':>{col_w}}")
        print("  " + "-" * (10 + 7 + col_w * 2 + 10))
        for ts, grp in df.groupby("train_size"):
            print(
                f"  {ts:>10}  {grp[n_col].mean():>7.1f}  "
                f"{_fmt(grp[vs_icl_col]):>{col_w}}  "
                f"{_fmt(grp[vs_gt_col]):>{col_w}}"
            )
        # pooled row
        print(
            f"  {'(pooled)':>10}  {df[n_col].mean():>7.1f}  "
            f"{_fmt(df[vs_icl_col]):>{col_w}}  "
            f"{_fmt(df[vs_gt_col]):>{col_w}}"
        )

    print(f"\n  ICL accuracy (probe-test subset): {_fmt(df['icl_acc'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Population agreement analysis: meta-probe vs ICL predictions."
    )
    # single-layer mode
    parser.add_argument(
        "--split_dir", type=str, default=None,
        help="Path to a single probe_on_icl_splits_labellayer* directory.",
    )
    parser.add_argument("--label_layer", type=int, default=4)
    # multi-layer mode
    parser.add_argument(
        "--base_dir", type=str, default=None,
        help="Parent directory containing probe_on_icl_splits_labellayer* subdirs.",
    )
    parser.add_argument(
        "--label_layers", type=int, nargs="+", default=None,
        help="One or more label layer indices (used with --base_dir).",
    )
    parser.add_argument("--hidden_layer", type=int, default=0)
    args = parser.parse_args()

    # ── resolve (split_dir, label_layer) pairs ────────────────────────────────
    if args.base_dir is not None and args.label_layers is not None:
        base = Path(args.base_dir)
        jobs = [
            (base / f"probe_on_icl_splits_labellayer{ll}", ll)
            for ll in args.label_layers
        ]
    elif args.split_dir is not None:
        jobs = [(Path(args.split_dir), args.label_layer)]
    else:
        parser.error("Provide either --split_dir or both --base_dir and --label_layers.")

    all_rows: list[dict] = []

    for split_dir, ll in jobs:
        print(f"\n{'='*60}")
        print(f"=== label_layer={ll}  hidden_layer={args.hidden_layer} ===")
        print(f"{'='*60}")

        try:
            records = load_split_records(split_dir, args.hidden_layer, ll)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue

        rows = [r for rec in records if (r := analyse_record(rec)) is not None]
        if not rows:
            print("  No valid records found.")
            continue

        df_ll = pd.DataFrame(rows)
        df_ll["label_layer"] = ll

        print(f"\nTotal splits analysed : {len(df_ll)}")
        print_summary(df_ll)

        out_path = split_dir / f"icl_probe_agreement_hl{args.hidden_layer}.csv"
        df_ll.to_csv(out_path, index=False)
        print(f"\nPer-split results saved to {out_path}")

        all_rows.extend(rows)

    # ── cross-layer aggregate (only meaningful when >1 layer) ─────────────────
    if len(jobs) > 1 and all_rows:
        print(f"\n{'='*60}")
        print(f"=== AGGREGATE across all label layers ===")
        print(f"{'='*60}")
        df_all = pd.DataFrame(all_rows)
        print(f"\nTotal splits analysed : {len(df_all)}")
        print_summary(df_all)


if __name__ == "__main__":
    main()
