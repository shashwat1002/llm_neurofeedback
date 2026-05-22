import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import expit
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from configs.settings import SELECTED_LAYERS
from utils import load_exp_cfg, load_saved_data
from plotter import set_mpl, PLOT_PARAMS, get_pc_colors
from tqdm import tqdm


def binary_ce_loss(true_scores, prob_scores, eps=1e-8):
    """Compute the binary cross entropy loss for a set of predictions.
    :param true_scores: array of ground truth values (0 or 1)
    :param prob_scores: array of probabilities (between 0 and 1)
    :param eps: a small number to avoid log(0)
    """
    true_scores = np.array(true_scores)
    prob_scores = np.array(prob_scores)
    # Apply the binary cross entropy loss formula elementwise
    ce = - (true_scores * np.log(prob_scores + eps) + (1 - true_scores) * np.log(1 - prob_scores + eps))
    return ce


def get_loss_and_acc(model_s, n_layers, pcs, save_dir, n_exp=100, n_sample=600, randomized=False):
    acc = np.zeros((len(pcs), n_layers, n_exp, n_sample))  # add one for lr
    ce_loss = np.zeros((len(pcs), n_layers, n_exp, n_sample))  # add one for lr

    for i_pc, pc_number in enumerate(pcs):
        cfg_s = load_exp_cfg(model_s, pc_number=pc_number)
        if pc_number == -1:  # for lr
            file_name = 'lr'
        else:
            file_name = f'{cfg_s.clf}_{cfg_s.clf_name}'
        if randomized:
            file_name += '_randomized'
        print("loading scores for", file_name)
        all_scores = load_saved_data(f"clf_{file_name}", save_dir, experiment='predict_hidden_mean')
        all_scores['est_correct'] = all_scores.apply(lambda row: np.array(row['all_example_est_scores']) == np.array(row['all_example_true_scores']), axis=1)
        # from logit to pr
        all_scores['prob'] = all_scores['all_example_est_scores_logitdiff'].apply(lambda x: expit(np.array(x)))
        # from pr to loss
        all_scores['ce_loss'] = all_scores.apply(lambda row: binary_ce_loss(row['all_example_true_scores'], row['prob']), axis=1)

        layers = sorted(all_scores['layer'].unique())
        experiments = sorted(all_scores['experiment'].unique())
        n_true_examples = len(all_scores.iloc[0]['all_example_true_scores'])

        for _, row in tqdm(all_scores.iterrows(), total=len(all_scores)):  # different layers
            i_layer = layers.index(row['layer'])
            i_exp = experiments.index(row['experiment'])
            if i_exp >= n_exp:
                continue
            acc[i_pc, i_layer, i_exp, :n_true_examples] = row['est_correct']
            ce_loss[i_pc, i_layer, i_exp, :n_true_examples] = row['ce_loss']

    print((ce_loss == 0).sum(), 'of data are all zeros')
    return acc, ce_loss


def plot_prediction_performance(acc, ce_loss, pcs, save_dir, colors, n_sample=600):
    acc_mean = np.mean(acc, axis=(1, 2))  # mean over layers and experiments
    acc_mean = gaussian_filter1d(acc_mean, 1, axis=1)
    acc_sem = np.std(acc, axis=(1, 2), ddof=1) / np.sqrt(acc.shape[1] * acc.shape[2])
    acc_sem = gaussian_filter1d(acc_sem, 1, axis=1)
    ce_loss_mean = np.mean(ce_loss, axis=(1, 2))
    ce_loss_mean = gaussian_filter1d(ce_loss_mean, 1, axis=1)
    ce_loss_std = np.std(ce_loss, axis=(1, 2), ddof=1) / np.sqrt(ce_loss.shape[1] * ce_loss.shape[2])
    ce_loss_std = gaussian_filter1d(ce_loss_std, 1, axis=1)


    fig, axes = plt.subplots(1, 2, figsize=(5, 2), dpi=300)
    for i, pc_number in enumerate(pcs):
        color = colors[i]
        axes[0].plot(np.arange(1, 1 + n_sample), acc_mean[i], alpha=0.9, color=color, linewidth=0.5)
        axes[1].plot(np.arange(1, 1 + n_sample), ce_loss_mean[i], alpha=0.9, color=color, linewidth=0.5)
        axes[0].fill_between(np.arange(1, 1 + n_sample), acc_mean[i] - acc_sem[i], acc_mean[i] + acc_sem[i], alpha=0.2, color=color)
        axes[1].fill_between(np.arange(1, 1 + n_sample), ce_loss_mean[i] - ce_loss_std[i], ce_loss_mean[i] + ce_loss_std[i], alpha=0.2, color=color)

    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('Cross-entropy')
    axes[0].set_ylim(0.5, 1)
    axes[1].set_ylim(0, 1.2)
    for j in range(2):
        axes[j].set_xlabel('# Examples')

    legend_elements = []
    for i, pc_number in enumerate(pcs):
        if pc_number == -1:
            legend_elements.append(Line2D([0], [0], color=colors[i], lw=1.5, label='LR'))
        else:
            legend_elements.append(Line2D([0], [0], color=colors[i], lw=1.5, label=f'PC{pcs[i]}'))

    plt.tight_layout(rect=(0, 0, 0.85, 1))
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.82, 0.6), bbox_transform=fig.transFigure)
    plt.savefig(f'{save_dir}/perf.{fig_format}', **PLOT_PARAMS)
    plt.close()


def plot_layers_prediction_performance(acc, pcs, fig_dir, colors, n_sample):
    acc_mean = np.mean(acc, axis=2)  # mean over layers and experiments
    acc_mean = gaussian_filter1d(acc_mean, 1, axis=-1)
    n_cols = 6

    fig, ax = plt.subplots(6, n_cols, figsize=(6, 6), dpi=300, sharey=True, sharex=True)
    for i, pc_number in enumerate(pcs):
        color = colors[i]
        for j in range(n_layers):
            ax[j // n_cols, j % n_cols].plot(np.arange(1, 1 + n_sample), acc_mean[i, j], alpha=0.9, color=color,
                                             linewidth=0.5)
            ax[j // n_cols, j % n_cols].set_ylim(0.45, 1)
            ax[j // n_cols, j % n_cols].set_xlim(0.9, n_sample)
            ax[j // n_cols, j % n_cols].set_title(f'Layer {j}', fontsize=7)
            if i == 0:
                ax[j // n_cols, j % n_cols].set_ylabel('Accuracy', fontsize=7)

    for idx in range(n_layers, 6 * n_cols):
        row, col = idx // n_cols, idx % n_cols
        ax[row, col].axis('off')

    legend_elements = []
    for i, pc_number in enumerate(pcs):
        if pc_number == -1:
            legend_elements.append(Line2D([0], [0], color=colors[i], lw=1.5, label='LR'))
        else:
            legend_elements.append(Line2D([0], [0], color=colors[i], lw=1.5, label=f'PC{pcs[i]}'))
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.7, 0.1), ncol=2, fontsize=8,
               bbox_transform=fig.transFigure)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/layers_perf.{fig_format}', **PLOT_PARAMS)
    plt.close()



def plot_prediction_performance_both(acc1, acc2, ce_loss1, ce_loss2, pcs, save_dir, colors1, colors2, fig_format, dataset_labels=('Original', 'Relabeled'), n_sample=600):
    # Helper to avoid repeating the mean, SEM, and filtering calculations
    def _get_smoothed_stats(data):
        mean = np.mean(data, axis=(1, 2))
        mean = gaussian_filter1d(mean, 1, axis=1)
        sem = np.std(data, axis=(1, 2), ddof=1) / np.sqrt(data.shape[1] * data.shape[2])
        sem = gaussian_filter1d(sem, 1, axis=1)
        return mean, sem

    # Process all four inputs
    acc1_mean, acc1_sem = _get_smoothed_stats(acc1)
    acc2_mean, acc2_sem = _get_smoothed_stats(acc2)
    ce1_mean, ce1_std = _get_smoothed_stats(ce_loss1)
    ce2_mean, ce2_std = _get_smoothed_stats(ce_loss2)

    # Group them for easy iteration
    acc_means = [acc1_mean, acc2_mean]
    acc_sems = [acc1_sem, acc2_sem]
    ce_means = [ce1_mean, ce2_mean]
    ce_stds = [ce1_std, ce2_std]
    
    # Store both color lists and define linestyles
    color_sets = [colors1, colors2]
    linestyles = ['-', '--'] 

    fig, axes = plt.subplots(1, 2, figsize=(5, 2), dpi=300)
    x_axis = np.arange(1, 1 + n_sample)

    # Loop over both datasets
    for k in range(2):
        ls = linestyles[k]
        current_colors = color_sets[k]
        
        # Loop over principal components
        for i, pc_number in enumerate(pcs):
            color = current_colors[i]
            
            # Plot lines
            axes[0].plot(x_axis, acc_means[k][i], alpha=0.9, color=color, linestyle=ls, linewidth=1.0)
            axes[1].plot(x_axis, ce_means[k][i], alpha=0.9, color=color, linestyle=ls, linewidth=1.0)
            
            # Plot fill bounds 
            axes[0].fill_between(x_axis, acc_means[k][i] - acc_sems[k][i], acc_means[k][i] + acc_sems[k][i], alpha=0.15, color=color)
            axes[1].fill_between(x_axis, ce_means[k][i] - ce_stds[k][i], ce_means[k][i] + ce_stds[k][i], alpha=0.15, color=color)

    # Formatting axes
    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('Cross-entropy')
    axes[0].set_ylim(0.5, 1)
    axes[1].set_ylim(0, 1.2)
    for j in range(2):
        axes[j].set_xlabel('# Examples')

    # Build Legend
    legend_elements = []
    
    # Create explicit legend entries for every line plotted
    for k in range(2):
        for i, pc_number in enumerate(pcs):
            pc_label = 'LR' if pc_number == -1 else f'PC{pcs[i]}'
            full_label = f'{dataset_labels[k]} ({pc_label})'
            
            legend_elements.append(Line2D(
                [0], [0], 
                color=color_sets[k][i], 
                lw=1.5, 
                linestyle=linestyles[k], 
                label=full_label
            ))

    # Adjust layout. We leave room on the right for the legend.
    plt.tight_layout(rect=(0, 0, 0.75, 1))
    
    # Added ncol=1 (or you can change to 2) depending on how many PCs you have so it doesn't overflow
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.76, 0.5), bbox_transform=fig.transFigure, fontsize=8)
    
    plt.savefig(f'{save_dir}/perf.{fig_format}', **PLOT_PARAMS)
    plt.close()

def plot_prediction_performance_acc_only(acc1, acc2, pcs, save_dir, colors1, colors2, fig_format, dataset_labels=('Original', 'Relabeled'), n_sample=600):
    def _get_smoothed_stats(data):
        mean = np.mean(data, axis=(1, 2))
        mean = gaussian_filter1d(mean, 1, axis=1)
        sem = np.std(data, axis=(1, 2), ddof=1) / np.sqrt(data.shape[1] * data.shape[2])
        sem = gaussian_filter1d(sem, 1, axis=1)
        return mean, sem

    acc1_mean, acc1_sem = _get_smoothed_stats(acc1)
    acc2_mean, acc2_sem = _get_smoothed_stats(acc2)

    acc_means = [acc1_mean, acc2_mean]
    acc_sems = [acc1_sem, acc2_sem]
    color_sets = [colors1, colors2]
    linestyles = ['-', '--']

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)
    x_axis = np.arange(1, 1 + n_sample)

    for k in range(2):
        ls = linestyles[k]
        current_colors = color_sets[k]
        for i, pc_number in enumerate(pcs):
            color = current_colors[i]
            ax.plot(x_axis, acc_means[k][i], alpha=0.9, color=color, linestyle=ls, linewidth=1.0)
            ax.fill_between(x_axis, acc_means[k][i] - acc_sems[k][i], acc_means[k][i] + acc_sems[k][i], alpha=0.15, color=color)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('# Examples')
    ax.set_ylim(0.4, 1)
    ax.axhline(0.5, color='black', linestyle=':', linewidth=0.8, label='random baseline')
    ax.text(x_axis[-1], 0.5, 'random baseline', va='bottom', ha='right', fontsize=6, color='black')

    legend_elements = []
    for k in range(2):
        for i, pc_number in enumerate(pcs):
            pc_label = 'LR' if pc_number == -1 else f'PC{pcs[i]}'
            full_label = f'{dataset_labels[k]} ({pc_label})'
            legend_elements.append(Line2D([0], [0], color=color_sets[k][i], lw=1.5, linestyle=linestyles[k], label=full_label))

    plt.tight_layout(rect=(0, 0, 1, 0.78))
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure, fontsize=8, ncol=2)
    plt.savefig(f'{save_dir}/perf_acc_only.{fig_format}', **PLOT_PARAMS)
    plt.close()


def plot_prediction_performance_acc_only_sparse(acc1, acc2, pcs, save_dir, colors1, colors2, fig_format, rate=50, dataset_labels=('Original', 'Relabeled'), n_sample=600):
    def _get_smoothed_stats(data):
        mean = np.mean(data, axis=(1, 2))
        sem = np.std(data, axis=(1, 2), ddof=1) / np.sqrt(data.shape[1] * data.shape[2])
        return mean, sem

    acc1_mean, acc1_sem = _get_smoothed_stats(acc1)
    acc2_mean, acc2_sem = _get_smoothed_stats(acc2)

    acc_means = [acc1_mean, acc2_mean]
    acc_sems = [acc1_sem, acc2_sem]
    color_sets = [colors1, colors2]
    markers = ['o', 's']

    x_all = np.arange(1, 1 + n_sample)
    x_sparse = x_all[(x_all % rate) == 0]

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)

    for k in range(2):
        current_colors = color_sets[k]
        marker = markers[k]
        for i, pc_number in enumerate(pcs):
            color = current_colors[i]
            y_sparse = acc_means[k][i][x_sparse - 1]
            sem_sparse = acc_sems[k][i][x_sparse - 1]
            ax.scatter(x_sparse, y_sparse, color=color, marker=marker, s=8, alpha=0.9, zorder=3)
            ax.errorbar(x_sparse, y_sparse, yerr=sem_sparse, fmt='none', color=color, alpha=0.7, linewidth=1.0, capsize=2.5)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('# Examples')
    ax.set_ylim(0.4, 1)
    ax.axhline(0.5, color='black', linestyle=':', linewidth=0.8)
    ax.text(x_sparse[-1], 0.5, 'random baseline', va='bottom', ha='right', fontsize=6, color='black')

    legend_elements = []
    for k in range(2):
        for i, pc_number in enumerate(pcs):
            pc_label = 'LR' if pc_number == -1 else f'PC{pcs[i]}'
            full_label = f'{dataset_labels[k]} ({pc_label})'
            legend_elements.append(Line2D([0], [0], color=color_sets[k][i], lw=0, marker=markers[k], markersize=4, label=full_label))

    plt.tight_layout(rect=(0, 0, 1, 0.78))
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure, fontsize=8, ncol=2)
    plt.savefig(f'{save_dir}/perf_acc_only_sparse.{fig_format}', **PLOT_PARAMS)
    plt.close()


if __name__ == "__main__":
    set_mpl()
    fig_format = 'svg'
    dataset_name, label_name = "commonsense", "labels"
    model_s = "llama3.1_8b"  # model generate score: "llama3.1_8b" or "qwen2.5_7b" or "llama3.1_70b" or "qwen2.5_72b"
    model_e = "llama3.1_8b"  # model run prediction exp: "llama3.1_8b" or "qwen2.5_7b" or "llama3.1_70b" or "qwen2.5_72b"
    randomized = False


    cfg_s = load_exp_cfg(model_s)
    cfg_e = load_exp_cfg(model_e)
    save_dir = Path("results") / (cfg_s.model_name.replace("/", "_") + '-'+ cfg_e.model_name.replace("/", "_")) / dataset_name
    fig_dir = f'{save_dir}/prediction'

    if randomized:
        fig_dir = fig_dir + '_randomized'

    os.makedirs(fig_dir, exist_ok=True)
    n_train_examples = cfg_s.n_train_examples
    n_exp = 100
    n_sample = 500
    n_layers = SELECTED_LAYERS[model_s][-1] + 1

    pcs = cfg_s.all_pc_exp
    pcs = pcs[:-1]
    colors = get_pc_colors(len(pcs))
    pcs = [-1] + pcs  # add one for lr
    colors = ['#9e0000'] + colors  # dark red for lr

    acc, ce_loss = get_loss_and_acc(model_s, n_layers, pcs, save_dir, n_exp=n_exp, n_sample=n_sample, randomized=randomized)
    plot_prediction_performance(acc, ce_loss, pcs, fig_dir, colors, n_sample=n_sample)
    plot_layers_prediction_performance(acc, pcs, fig_dir, colors, n_sample=n_sample)
