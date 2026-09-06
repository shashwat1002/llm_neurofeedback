import csv
import glob
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.special import expit
from scipy.ndimage import gaussian_filter1d
from configs.settings import SELECTED_LAYERS
from utils import load_exp_cfg, load_saved_data
from plotter import set_mpl, PLOT_PARAMS, get_pc_colors
from plot_pred import get_loss_and_acc


def _lighten_color(color, amount=0.5):
    """Return a lighter version of *color* by blending toward white."""
    rgba = np.array(mcolors.to_rgba(color))
    rgba[:3] = 1 - (1 - rgba[:3]) * (1 - amount)
    return tuple(rgba)


def load_probe_results(probe_dir, pcs):
    """Load probe CSV files and return mean accuracy per (pc, train_size) averaged over layers.

    Returns:
        probe_train_sizes: sorted list of unique train_size values
        probe_acc_mean: array of shape (len(pcs),) with mean accuracy per pc,
                        each element is an array over train_sizes
        probe_acc_sem: same shape, standard error of the mean
    """
    print("load probe results")
    probe_dir = Path(probe_dir)
    all_train_sizes = None
    probe_acc_mean = []
    probe_acc_sem = []

    for pc_number in pcs:
        pattern = str(
            probe_dir
            / f"hidden_mean_pcascore_pc{pc_number}_example_scores.probe_results.csv"
        )
        matches = glob.glob(pattern)
        if not matches:
            probe_acc_mean.append(None)
            probe_acc_sem.append(None)
            continue

        rows = []
        with open(matches[0], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    (
                        int(row["label_layer"]),
                        int(row["train_size"]),
                        float(row["accuracy"]),
                    )
                )

        train_sizes = sorted(set(r[1] for r in rows))
        if all_train_sizes is None:
            all_train_sizes = train_sizes

        # Average accuracy across all layers for each train_size.
        # SEM is computed across layer means (not pooled over all reps×layers),
        # so it reflects variability across layers rather than shrinking to
        # near-zero from dividing by sqrt(n_layers * n_reps).
        label_layers = sorted(set(r[0] for r in rows))
        means = []
        sems = []
        for ts in train_sizes:
            layer_means = [
                np.mean([r[2] for r in rows if r[1] == ts and r[0] == ll])
                for ll in label_layers
            ]
            means.append(np.mean(layer_means))
            sems.append(np.std(layer_means, ddof=1) / np.sqrt(len(layer_means)))
        probe_acc_mean.append(np.array(means))
        probe_acc_sem.append(np.array(sems))

    return all_train_sizes, probe_acc_mean, probe_acc_sem


def plot_prediction_performance_with_probe(
    acc,
    ce_loss,
    pcs,
    save_dir,
    colors,
    probe_train_sizes,
    probe_acc_mean,
    probe_acc_sem,
    fig_format="svg",
    n_sample=600,
    model_name=None,
):
    acc_mean = np.mean(acc, axis=(1, 2))
    acc_mean = gaussian_filter1d(acc_mean, 1, axis=1)
    acc_sem = np.std(acc, axis=(1, 2), ddof=1) / np.sqrt(acc.shape[1] * acc.shape[2])
    acc_sem = gaussian_filter1d(acc_sem, 1, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)
    x_axis = np.arange(1, 1 + n_sample)

    for i, pc_number in enumerate(pcs):
        color = colors[i]
        ax.plot(x_axis, acc_mean[i], alpha=0.9, color=color, linewidth=0.5)
        ax.fill_between(
            x_axis,
            acc_mean[i] - acc_sem[i],
            acc_mean[i] + acc_sem[i],
            alpha=0.2,
            color=color,
        )

    # Overlay probe results as markers on the accuracy plot
    # Only PCs that have a valid probe CSV (skip -1 / LR and missing entries)
    for i, pc_number in enumerate(pcs):
        if pc_number == -1:
            continue
        if probe_acc_mean[i] is None or probe_train_sizes is None:
            continue
        color = colors[i]
        ax.errorbar(
            probe_train_sizes,
            probe_acc_mean[i],
            yerr=probe_acc_sem[i],
            fmt="o",
            color=color,
            markersize=4,
            linewidth=0.8,
            capsize=2,
            zorder=5,
            markeredgecolor="black",
            markeredgewidth=0.7,
        )

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1)
    ax.set_xlabel("# Examples")

    if model_name is not None:
        model_name_sanitized = model_name.split("/")[-1]
        ax.set_title(model_name_sanitized, fontsize=8)

    legend_elements = []
    for i, pc_number in enumerate(pcs):
        if pc_number == -1:
            legend_elements.append(
                Line2D([0], [0], color=colors[i], lw=1.5, label="LR")
            )
        else:
            legend_elements.append(
                Line2D([0], [0], color=colors[i], lw=1.5, label=f"PC{pc_number}")
            )
    # Add a generic marker entry to explain the probe dots
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            lw=0,
            markersize=4,
            label="Probe (avg layers)",
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
    )

    plt.tight_layout(rect=(0, 0, 1, 0.78))
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=fig.transFigure,
        fontsize=8,
        ncol=2,
    )
    plt.savefig(f"{save_dir}/perf_with_probe.{fig_format}", **PLOT_PARAMS)
    plt.close()


def plot_matched_probe_acc(
    acc,
    pcs,
    save_dir,
    colors,
    probe_train_sizes,
    probe_acc_mean,
    probe_acc_sem,
    fig_format="svg",
    model_name=None,
):
    """Plot acc and probe_acc_mean only at the x-positions where probe data exists.

    For each PC, extracts acc_mean at the exact train_size indices from probe_train_sizes
    so both series are plotted at the same x-positions.
    """
    # Average over layers first, then compute SEM across experiments.
    # Dividing by sqrt(n_layers * n_exp) makes SEM ~0.009 (sub-pixel); using
    # only n_exp as denominator gives ~5x larger, visible error bars.
    acc_per_exp = np.mean(acc, axis=1)  # (n_pcs, n_exp, n_sample)
    acc_mean = np.mean(acc_per_exp, axis=1)  # (n_pcs, n_sample)
    acc_sem = np.std(acc_per_exp, axis=1, ddof=1) / np.sqrt(acc.shape[2])  # sqrt(n_exp)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)

    # Compute jitter offsets so acc and probe markers never overlap
    n_valid = sum(
        1 for i, pc in enumerate(pcs) if pc != -1 and probe_acc_mean[i] is not None
    )
    jitter_offsets = np.linspace(-0.02, 0.02, max(n_valid * 2, 2))
    jitter_idx = 0

    for i, pc_number in enumerate(pcs):
        if pc_number == -1:
            continue
        if probe_acc_mean[i] is None or probe_train_sizes is None:
            continue

        color = colors[i]
        probe_color = _lighten_color(color, amount=0.5)
        xs = np.array(probe_train_sizes, dtype=float)
        acc_at_probe = acc_mean[i][np.array(probe_train_sizes) - 1]
        acc_sem_at_probe = acc_sem[i][np.array(probe_train_sizes) - 1]

        # Relative jitter so spacing looks even across wide x-ranges
        xs_acc = xs * (1 + jitter_offsets[jitter_idx])
        xs_probe = xs * (1 + jitter_offsets[jitter_idx + 1])
        jitter_idx += 2

        ax.errorbar(
            xs_acc,
            acc_at_probe,
            yerr=acc_sem_at_probe,
            fmt="s",
            color=color,
            markersize=4,
            linewidth=0.8,
            capsize=2,
            zorder=4,
            markeredgecolor="black",
            markeredgewidth=0.7,
            label=f"PC{pc_number} acc",
        )
        ax.errorbar(
            xs_probe,
            probe_acc_mean[i],
            yerr=probe_acc_sem[i],
            fmt="o",
            color=probe_color,
            markersize=4,
            linewidth=0.8,
            capsize=2,
            zorder=5,
            markeredgecolor="black",
            markeredgewidth=0.7,
            label=f"PC{pc_number} probe",
        )

    # label the grey line at 0.5 as "random chance" in the legend
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, zorder=1)
    

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1)
    ax.set_xlabel("# Examples")

    # if model_name is not None:
    #     ax.set_title(model_name.split("/")[-1], fontsize=8)

    legend_elements = []
    for i, pc_number in enumerate(pcs):
        if pc_number == -1 or probe_acc_mean[i] is None:
            continue
        color = colors[i]
        probe_color = _lighten_color(color, amount=0.5)
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color=color,
                lw=0,
                markersize=4,
                label=f"Model Prediction PC{pc_number}",
                markeredgecolor="black",
                markeredgewidth=0.7,
            )
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=probe_color,
                lw=0,
                markersize=4,
                label=f"Layer-0 probe prediction for PC{pc_number}",
                markeredgecolor="black",
                markeredgewidth=0.7,
            )
        )
    legend_elements.append(
        Line2D(
            [0], [0], color="grey", linestyle="--", linewidth=0.8, label="Random Chance"
        )
    )

    plt.tight_layout(rect=(0, 0, 1, 0.78))
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=fig.transFigure,
        fontsize=8,
        ncol=2,
    )
    plt.savefig(f"{save_dir}/matched_probe_acc.{fig_format}", **PLOT_PARAMS)
    plt.close()


if __name__ == "__main__":
    set_mpl()
    fig_format = "svg"
    dataset_name = "commonsense"
    model_s = "llama3.1_8b"
    model_e = "llama3.1_8b"
    randomized = False

    cfg_s = load_exp_cfg(model_s)
    cfg_e = load_exp_cfg(model_e)
    save_dir = (
        Path("results")
        / (
            cfg_s.model_name.replace("/", "_")
            + "-"
            + cfg_e.model_name.replace("/", "_")
        )
        / dataset_name
    )
    fig_dir = f"{save_dir}/prediction"

    if randomized:
        fig_dir = fig_dir + "_randomized"

    os.makedirs(fig_dir, exist_ok=True)

    n_exp = 100
    n_sample = 500
    n_layers = SELECTED_LAYERS[model_s][-1] + 1

    pcs = cfg_s.all_pc_exp
    pcs = [1]
    colors = get_pc_colors(len(pcs))

    # The probe CSV directory — where per-layer probe results are stored
    probe_dir = Path("results") / cfg_s.model_name.replace("/", "_") / dataset_name

    acc, ce_loss = get_loss_and_acc(
        model_s,
        n_layers,
        pcs,
        save_dir,
        n_exp=n_exp,
        n_sample=n_sample,
        randomized=randomized,
    )

    # pcs list includes -1 (LR) at index 0; pass only the real PC numbers for probe loading
    pc_numbers_for_probe = [pc if pc != -1 else None for pc in pcs]
    real_pcs = [pc for pc in pcs if pc != -1]
    probe_train_sizes, probe_means_list, probe_sems_list = load_probe_results(
        probe_dir, real_pcs
    )

    # Rebuild probe lists aligned with full pcs list (index 0 = LR → None)
    probe_acc_mean = probe_means_list
    probe_acc_sem = probe_sems_list

    # plot_prediction_performance_with_probe(
    #     acc, ce_loss, pcs, fig_dir, colors,
    #     probe_train_sizes, probe_acc_mean, probe_acc_sem,
    #     fig_format=fig_format, n_sample=n_sample,
    #     model_name=cfg_s.model_name,
    # )

    plot_matched_probe_acc(
        acc,
        pcs,
        fig_dir,
        colors,
        probe_train_sizes,
        probe_acc_mean,
        probe_acc_sem,
        fig_format=fig_format,
        model_name=cfg_s.model_name,
    )
