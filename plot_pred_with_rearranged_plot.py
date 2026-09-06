import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import expit
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from configs.settings import SELECTED_LAYERS
from utils import load_exp_cfg, load_saved_data
from plotter import set_mpl, PLOT_PARAMS
from plot_pred import (
    plot_prediction_performance,
    plot_layers_prediction_performance,
    get_loss_and_acc,
    plot_prediction_performance_both,
    plot_prediction_performance_acc_only,
    plot_prediction_performance_acc_only_sparse
)


if __name__ == "__main__":
    set_mpl()
    fig_format = "svg"
    dataset_name, label_name = "sst2", "labels"
    model_s = "qwen2.5_7b_1m"  # model generate score: "llama3.1_8b" or "qwen2.5_7b" or "llama3.1_70b" or "qwen2.5_72b"
    model_e = "qwen2.5_7b_1m"  # model run prediction exp: "llama3.1_8b" or "qwen2.5_7b" or "llama3.1_70b" or "qwen2.5_72b"
    randomized = True

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

    fig_dir = fig_dir + "_lr_plus_ranomized"

    os.makedirs(fig_dir, exist_ok=True)
    n_train_examples = cfg_s.n_train_examples
    cmap = plt.get_cmap("viridis")
    n_exp = 30
    n_sample = 5000
    n_layers = SELECTED_LAYERS[model_s][-1] + 1

    pcs = []
    pc_positions = np.linspace(1, 0, len(pcs))
    colors = [cmap(p) for p in pc_positions]
    colors2 = [cmap(p) for p in pc_positions]
    colors2.reverse()  # reverse the order of colors for the second plot
    pcs = [-1] + pcs  # add one for lr
    colors = ["red"] + colors  # use darkred for lr
    colors2 = ["grey"] + colors2  # use darkyellow for lr

    acc, ce_loss = get_loss_and_acc(
        model_s,
        n_layers,
        pcs,
        save_dir,
        n_exp=n_exp,
        n_sample=n_sample,
        randomized=False,
    )
    # breakpoint()
    acc_rand, ce_loss_rand = get_loss_and_acc(
        model_s,
        n_layers,
        pcs,
        save_dir,
        n_exp=n_exp,
        n_sample=n_sample,
        randomized=True,
    )
    # breakpoint()

    # plot_prediction_performance_acc_only(
    #     acc,
    #     acc_rand,
    #     pcs,
    #     fig_dir,
    #     colors,
    #     colors2,
    #     fig_format=fig_format,
    #     dataset_labels=("Original", "Randomized"),
    #     n_sample=n_sample,
    # )

    plot_prediction_performance_acc_only_sparse(
        acc,
        acc_rand,
        pcs,
        fig_dir,
        colors,
        colors2,
        fig_format=fig_format,
        rate=500,
        dataset_labels=("Original", "Relabeled"),
        n_sample=n_sample,
    )
