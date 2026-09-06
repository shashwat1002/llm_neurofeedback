import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler


PLOT_PARAMS = {
    "dpi": 300,
    "bbox_inches": 'tight',
    "pad_inches": 0.1,
}

# Dark, colorblind-friendly palette (8 colors)
DARK_COLORS = [
    '#003f8e',  # dark blue
    '#9e0000',  # dark red
    '#005c3b',  # dark green
    '#7a3800',  # dark orange
    '#4b006e',  # dark purple
    '#006670',  # dark teal
    '#5c4000',  # dark gold/brown
    '#3d3d3d',  # dark grey
]


def get_pc_colors(n):
    """Return n dark colors from DARK_COLORS (cycling if n > 8)."""
    return [DARK_COLORS[i % len(DARK_COLORS)] for i in range(n)]


def set_mpl():
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['savefig.dpi'] = 600
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.prop_cycle'] = cycler(color=DARK_COLORS)


def plot_neural_classifier_accuracies(layers, train_accuracies, test_accuracies, axis_name, hidden_method, save_dir, size=3):
    """Plot train and test accuracies for each layer's classifier."""
    plt.figure(figsize=(size, size))
    plt.plot(layers, train_accuracies, label="Train")
    plt.plot(layers, test_accuracies, label="Test")
    plt.axhline(y=0.5, color='r', linestyle='--', label="Chance level")  # chance level
    plt.xlabel("Layer")
    plt.ylabel(f"{axis_name} Accuracy")
    plt.title(hidden_method)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"hidden_axis_accuracies_{axis_name}.pdf", **PLOT_PARAMS)
    plt.close()


def plot_neuro_scores_distribution(layers, scores, save_file):
    plt.figure(figsize=(12, 12))
    n_layers = len(layers)
    n_row = int(n_layers ** 0.5) + 1
    for layer in layers:
        plt.subplot(n_row, n_row, layer + 1)
        score = scores[layer]
        plt.hist(score, bins=20)
        plt.title(layer)
    plt.tight_layout()
    plt.savefig(f'{save_file}.png', **PLOT_PARAMS)
    plt.close()
