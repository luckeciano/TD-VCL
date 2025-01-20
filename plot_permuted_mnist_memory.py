import argparse
import torch
from benchmarks import PermutedMNIST
from trainers import OnlineMLETrainer, BatchMLETrainer
from modules import MultiHeadMLP
import numpy as np
import pandas as pd
import random
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_utils import load_results


def main(args):
    sns.set_style("darkgrid")
    sns.set_context("paper")
    sns.set(font_scale=3.0)
    plot_dfs = []
    # Use seaborn to create the plot
    fig, axs = plt.subplots(1, 3, figsize=(10 * 3, 12))

    configs = [[(10, 60000), (10, 10000), (10, 1000), (10, 200), (10, 50), (10, 10)],
               [(5, 60000), (5, 10000), (5, 1000), (5, 200), (5, 50), (5, 10)],
               [(2, 60000), (2, 10000), (2, 1000), (2, 200), (2, 50), (2, 10)]]

    plot_dfs = load_results(filename="results/mnist_memory_online_mle.pkl")

    for idx, task_config in enumerate(configs):
       task_plot_dfs = load_results(f"results/mnist_memory_batch_mle_{idx}.pkl")
       plot_values(plot_dfs + task_plot_dfs, args.num_tasks, axs[idx], idx)

    
    plt.suptitle('PermutedMNIST: Replay Buffer Analysis')
    plt.tight_layout()
    plt.savefig('./iclr_permuted_mnist_memory.png', bbox_inches='tight')
    plt.savefig('./iclr_permuted_mnist_memory.pdf', bbox_inches='tight')
    plt.savefig('./iclr_permuted_mnist_memory.eps', bbox_inches='tight')


def generate_plt_dfs(results, method):
    df = pd.DataFrame(results)
    df.columns = df.columns + 1
    df_melted = df.melt(var_name='# tasks', value_name='Accuracy')
    df_melted['Method'] = [method] * len(df_melted)
    return df_melted

def plot_values(dfs, num_tasks, ax, idx):
    df_melted = pd.concat(dfs)

    sns.pointplot(x='# tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax)

    # Set the y-axis limits and ticks
    ax.set_ylim(0.3, 1.01)
    ax.set_yticks(np.arange(0.3, 1.01, 0.1))

    # Set the x-axis ticks
    ax.set_xticks(np.arange(0, num_tasks, 1))

    if idx != 0:
        ax.set_ylabel("")

    if idx != 1:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Number of Observed Tasks")

    # Get the handles and labels from the current legend
    handles, labels = ax.get_legend_handles_labels()

    # Find the index of "Online MLE" and move it to the end
    online_mle_index = labels.index("Online MLE")
    handles.append(handles.pop(online_mle_index))
    labels.append(labels.pop(online_mle_index))

    ax.legend(handles, labels, title='Method', loc='lower left', fontsize=23, title_fontsize=23)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Permuted MNIST.')
    parser.add_argument('--num_tasks', type=int, default=10,
                        help='The number of tasks.')
    parser.add_argument('--epochs_per_task', type=int, default=100,
                        help='The number of tasks.')
    parser.add_argument('--layers', type=str, default="[100, 100]",
                        help='The hidden layers.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='The number of tasks.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The learning rate.')
    parser.add_argument('--single_head', type=bool, default=True,
                        help='The learning rate.')
    parser.add_argument('--valid_ratio', type=float, default=0.01,
                        help='Train/Valid data ratio.')
    parser.add_argument('--enable_early_stopping', type=bool, default=True,
                        help='Whether to enable early stopping')
    parser.add_argument('--es_patience', type=int, default=5,
                        help='Early Stopping patience.')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='The number of experiment seeds.')
    
    

    args = parser.parse_args()
    main(args)
  