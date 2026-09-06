"""
Explore sentences and per-predictor predictions stored in probe_icl_combined.py
split caches.

Usage examples:

  # Show all sentences from one specific split
  python explore_predictions.py --splits_dir results/.../probe_icl_splits_layer16 \
      --layer 16 --train_size 200 --outer 0 --inner 0

  # Show only probe/ICL disagreements across all splits for a layer+train_size
  python explore_predictions.py --splits_dir results/.../probe_icl_splits_layer16 \
      --layer 16 --train_size 200 --show disagree

  # Show only ICL errors (ICL wrong, probe right)
  python explore_predictions.py --splits_dir results/.../probe_icl_splits_layer16 \
      --layer 16 --show icl_wrong

  # Dump to CSV instead of printing
  python explore_predictions.py --splits_dir results/.../probe_icl_splits_layer16 \
      --layer 16 --train_size 100 --csv out.csv
"""

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

from joblib import load


# ── helpers ───────────────────────────────────────────────────────────────────

LABEL_STR = {0: "low", 1: "high"}


def parse_split_filename(name: str) -> dict | None:
    m = re.fullmatch(
        r"outer(\d+)_inner(\d+)_size(\d+)_layer(\d+)\.pkl", name
    )
    if not m:
        return None
    return {
        "outer": int(m.group(1)),
        "inner": int(m.group(2)),
        "train_size": int(m.group(3)),
        "label_layer": int(m.group(4)),
    }


def load_splits(splits_dir: Path, layer=None, train_size=None, outer=None, inner=None):
    records = []
    for fname in sorted(os.listdir(splits_dir)):
        meta = parse_split_filename(fname)
        if meta is None:
            continue
        if layer is not None and meta["label_layer"] != layer:
            continue
        if train_size is not None and meta["train_size"] != train_size:
            continue
        if outer is not None and meta["outer"] != outer:
            continue
        if inner is not None and meta["inner"] != inner:
            continue
        rec = load(splits_dir / fname)
        records.append(rec)
    return records


def build_rows(records):
    """Flatten all records into a list of per-sentence dicts."""
    rows = []
    for rec in records:
        sents   = rec["test_sentences_icl"]
        true    = rec["true_labels_icl"]
        icl     = rec["icl_preds"]
        probe1  = rec["probe_preds_icl"]
        probe2  = rec["probe_preds2_icl"]
        meta = {
            "outer":       rec["outer_i"],
            "inner":       rec["inner_i"],
            "train_size":  rec["train_size"],
            "label_layer": rec["label_layer"],
        }
        for sent, t, ip, p1, p2 in zip(sents, true, icl, probe1, probe2):
            rows.append({
                **meta,
                "sentence":    sent,
                "true":        t,
                "icl":         ip,
                "probe1":      p1,
                "probe2":      p2,
                "icl_correct":    int(ip == t),
                "probe1_correct": int(p1 == t),
                "probe_icl_agree": int(p1 == ip),
            })
    return rows


def filter_rows(rows, show: str):
    if show == "all":
        return rows
    elif show == "disagree":
        return [r for r in rows if r["probe_icl_agree"] == 0]
    elif show == "agree":
        return [r for r in rows if r["probe_icl_agree"] == 1]
    elif show == "icl_wrong":
        return [r for r in rows if r["icl_correct"] == 0]
    elif show == "icl_right":
        return [r for r in rows if r["icl_correct"] == 1]
    elif show == "probe_wrong":
        return [r for r in rows if r["probe1_correct"] == 0]
    elif show == "both_wrong":
        return [r for r in rows if r["icl_correct"] == 0 and r["probe1_correct"] == 0]
    elif show == "only_icl_wrong":
        return [r for r in rows if r["icl_correct"] == 0 and r["probe1_correct"] == 1]
    elif show == "only_probe_wrong":
        return [r for r in rows if r["icl_correct"] == 1 and r["probe1_correct"] == 0]
    else:
        raise ValueError(f"Unknown --show value: {show!r}")


# ── display ───────────────────────────────────────────────────────────────────

SHOW_CHOICES = [
    "all", "disagree", "agree",
    "icl_wrong", "icl_right",
    "probe_wrong", "both_wrong",
    "only_icl_wrong", "only_probe_wrong",
]

TICK  = "✓"
CROSS = "✗"


def fmt_pred(pred, true):
    label = LABEL_STR.get(pred, str(pred))
    mark  = TICK if pred == true else CROSS
    return f"{label}({mark})"


def print_rows(rows, max_sent_width=90):
    if not rows:
        print("No rows match the current filters.")
        return

    sep = "-" * (max_sent_width + 50)
    header = (
        f"{'#':>5}  "
        f"{'L':>3}  {'N':>4}  {'O':>2}  {'I':>2}  "
        f"{'true':>6}  {'icl':>9}  {'p1':>9}  {'p2':>9}  "
        f"sentence"
    )
    print(sep)
    print(header)
    print(sep)

    for idx, r in enumerate(rows):
        sent = r["sentence"]
        if len(sent) > max_sent_width:
            sent = sent[:max_sent_width - 3] + "..."
        true_str = LABEL_STR.get(r["true"], str(r["true"]))
        print(
            f"{idx+1:>5}  "
            f"{r['label_layer']:>3}  {r['train_size']:>4}  "
            f"{r['outer']:>2}  {r['inner']:>2}  "
            f"{true_str:>6}  "
            f"{fmt_pred(r['icl'],    r['true']):>9}  "
            f"{fmt_pred(r['probe1'], r['true']):>9}  "
            f"{fmt_pred(r['probe2'], r['true']):>9}  "
            f"{sent}"
        )

    print(sep)
    n = len(rows)
    icl_acc   = sum(r["icl_correct"]    for r in rows) / n
    p1_acc    = sum(r["probe1_correct"] for r in rows) / n
    agree_rate = sum(r["probe_icl_agree"] for r in rows) / n
    print(
        f"  {n} rows shown  |  "
        f"icl_acc={icl_acc:.3f}  probe1_acc={p1_acc:.3f}  "
        f"agree_rate={agree_rate:.3f}"
    )


def write_dump(records, path: str, show: str, max_sent_width: int, limit: int | None):
    """Write all splits to a single text file, each split in its own section."""
    with open(path, "w") as f:
        total_written = 0
        for rec in records:
            rows = build_rows([rec])
            rows = filter_rows(rows, show)
            if not rows:
                continue

            ol = rec["outer_i"]
            il = rec["inner_i"]
            ts = rec["train_size"]
            ll = rec["label_layer"]
            kappa  = rec.get("kappa",         float("nan"))
            kappa2 = rec.get("kappa2",        float("nan"))
            kappa_ctrl = rec.get("kappa_control", float("nan"))
            ag1, ag2, ag12 = _raw_agreements(rec)

            sep  = "=" * (max_sent_width + 50)
            sep2 = "-" * (max_sent_width + 50)
            f.write(f"\n{sep}\n")
            f.write(
                f"  outer={ol}  inner={il}  train_size={ts}  label_layer={ll}\n"
            )
            f.write(
                f"  kappa(p1,icl)={kappa:.3f}  agree(p1,icl)={ag1:.3f}"
                f"  |  kappa(p2,icl)={kappa2:.3f}  agree(p2,icl)={ag2:.3f}"
                f"  |  kappa(p1,p2)={kappa_ctrl:.3f}  agree(p1,p2)={ag12:.3f}\n"
            )
            f.write(f"  probe1_acc={rec['probe_report']['accuracy']:.3f}"
                    f"  probe2_acc={rec['probe_report2']['accuracy']:.3f}"
                    f"  icl_acc={rec['model_report']['accuracy']:.3f}\n")
            f.write(f"{sep}\n")

            header = (
                f"{'#':>5}  {'true':>6}  {'icl':>9}  {'p1':>9}  {'p2':>9}  sentence"
            )
            f.write(header + "\n")
            f.write(sep2 + "\n")

            for idx, r in enumerate(rows):
                if limit is not None and total_written >= limit:
                    break
                sent = r["sentence"]
                if len(sent) > max_sent_width:
                    sent = sent[:max_sent_width - 3] + "..."
                true_str = LABEL_STR.get(r["true"], str(r["true"]))
                f.write(
                    f"{idx+1:>5}  "
                    f"{true_str:>6}  "
                    f"{fmt_pred(r['icl'],    r['true']):>9}  "
                    f"{fmt_pred(r['probe1'], r['true']):>9}  "
                    f"{fmt_pred(r['probe2'], r['true']):>9}  "
                    f"{sent}\n"
                )
                total_written += 1

            n = len(rows)
            icl_acc    = sum(r["icl_correct"]    for r in rows) / n
            p1_acc     = sum(r["probe1_correct"] for r in rows) / n
            agree_rate = sum(r["probe_icl_agree"] for r in rows) / n
            f.write(sep2 + "\n")
            f.write(
                f"  {n} rows  |  icl_acc={icl_acc:.3f}"
                f"  probe1_acc={p1_acc:.3f}  agree_rate={agree_rate:.3f}\n"
            )

            if limit is not None and total_written >= limit:
                f.write("\n[limit reached]\n")
                break

        sep = "=" * (max_sent_width + 50)
        f.write(f"\n{sep}\n")
        f.write(f"=== Summary across {len(records)} split(s) ===\n")
        for line in summary_lines(records):
            f.write(line + "\n")

        layers = sorted({r["label_layer"] for r in records})
        if len(layers) > 1:
            f.write(f"\n=== Layer-by-layer breakdown ===\n")
            for ll in layers:
                recs_ll = [r for r in records if r["label_layer"] == ll]
                f.write(f"\n  label_layer={ll}  ({len(recs_ll)} split(s))\n")
                for line in summary_lines(recs_ll):
                    f.write("  " + line + "\n")

    print(f"Wrote dump to {path}")


def _raw_agreements(rec: dict) -> tuple[float, float, float]:
    """Return (agree_p1_icl, agree_p2_icl, agree_p1_p2) for a single record."""
    p1  = rec["probe_preds_icl"]
    p2  = rec["probe_preds2_icl"]
    icl = rec["icl_preds"]
    n   = len(icl)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    a1  = sum(a == b for a, b in zip(p1,  icl)) / n
    a2  = sum(a == b for a, b in zip(p2,  icl)) / n
    a12 = sum(a == b for a, b in zip(p1,  p2))  / n
    return a1, a2, a12


def _mean_std(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / n) if n > 1 else 0.0
    return mu, sd


def summary_lines(records: list[dict]) -> list[str]:
    """Return lines describing mean ± std of all per-split quantities."""
    if not records:
        return ["  (no records)"]

    ag = [_raw_agreements(r) for r in records]
    quantities = {
        "probe1_acc":     [r["probe_report"]["accuracy"]          for r in records],
        "probe2_acc":     [r["probe_report2"]["accuracy"]         for r in records],
        "icl_acc":        [r["model_report"]["accuracy"]          for r in records],
        "kappa(p1,icl)":  [r.get("kappa",         float("nan"))  for r in records],
        "agree(p1,icl)":  [a[0] for a in ag],
        "kappa(p2,icl)":  [r.get("kappa2",        float("nan"))  for r in records],
        "agree(p2,icl)":  [a[1] for a in ag],
        "kappa(p1,p2)":   [r.get("kappa_control", float("nan"))  for r in records],
        "agree(p1,p2)":   [a[2] for a in ag],
    }

    lines = [f"  {'quantity':<20}  {'mean':>8}  {'std':>8}  {'max':>8}  {'n':>5}"]
    lines.append("  " + "-" * 58)
    for name, vals in quantities.items():
        clean = [v for v in vals if not math.isnan(v)]
        mu, sd = _mean_std(vals)
        mx = max(clean) if clean else float("nan")
        if name.startswith("kappa") or name.startswith("agree"):
            lines.append(f"  {name:<20}  {mu:>8.4f}  {sd:>8.4f}  {mx:>8.4f}  {len(clean):>5}")
        else:
            lines.append(f"  {name:<20}  {mu:>8.4f}  {sd:>8.4f}  {'':>8}  {len(clean):>5}")
    return lines


def print_summary(records: list[dict]):
    print(f"\n=== Summary across {len(records)} split(s) ===")
    for line in summary_lines(records):
        print(line)

    layers = sorted({r["label_layer"] for r in records})
    if len(layers) > 1:
        print(f"\n=== Layer-by-layer breakdown ===")
        for ll in layers:
            recs_ll = [r for r in records if r["label_layer"] == ll]
            print(f"\n  label_layer={ll}  ({len(recs_ll)} split(s))")
            for line in summary_lines(recs_ll):
                print("  " + line)


def write_csv(rows, path: str):
    if not rows:
        print("No rows to write.")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explore sentences and predictions from probe_icl_combined split caches.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--splits_dir", required=True,
        help="Path to probe_icl_splits_layer<N> directory.",
    )
    parser.add_argument("--layer",      type=int, default=None, help="Filter by label layer.")
    parser.add_argument("--train_size", type=int, default=None, help="Filter by train size.")
    parser.add_argument("--outer",      type=int, default=None, help="Filter by outer fold index.")
    parser.add_argument("--inner",      type=int, default=None, help="Filter by inner fold index.")
    parser.add_argument(
        "--show", default="all", choices=SHOW_CHOICES,
        help=(
            "Which rows to display:\n"
            "  all            - every sentence\n"
            "  disagree       - probe1 and ICL predict differently\n"
            "  agree          - probe1 and ICL predict the same\n"
            "  icl_wrong      - ICL prediction is wrong\n"
            "  icl_right      - ICL prediction is correct\n"
            "  probe_wrong    - probe1 prediction is wrong\n"
            "  both_wrong     - both ICL and probe1 are wrong\n"
            "  only_icl_wrong - ICL wrong but probe1 right\n"
            "  only_probe_wrong - probe1 wrong but ICL right\n"
        ),
    )
    parser.add_argument(
        "--csv", default=None, metavar="PATH",
        help="Write results to a CSV file instead of printing.",
    )
    parser.add_argument(
        "--dump", default=None, metavar="PATH",
        help="Write all splits to a single text file with a section header per split.",
    )
    parser.add_argument(
        "--max_width", type=int, default=90,
        help="Max character width for sentence column (default 90).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Show at most this many rows.",
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    if not splits_dir.is_dir():
        sys.exit(f"Not a directory: {splits_dir}")

    print(f"Loading splits from {splits_dir} ...", end=" ", flush=True)
    records = load_splits(
        splits_dir,
        layer=args.layer,
        train_size=args.train_size,
        outer=args.outer,
        inner=args.inner,
    )
    print(f"{len(records)} split file(s) loaded.")

    if not records:
        sys.exit("No matching split files found — check your filter arguments.")

    if args.dump:
        write_dump(records, args.dump, args.show, args.max_width, args.limit)
        return

    rows = build_rows(records)
    rows = filter_rows(rows, args.show)

    if args.limit is not None:
        rows = rows[:args.limit]

    if args.csv:
        write_csv(rows, args.csv)
    else:
        print_rows(rows, max_sent_width=args.max_width)
        print_summary(records)


if __name__ == "__main__":
    main()
