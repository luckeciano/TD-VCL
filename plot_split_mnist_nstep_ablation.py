import argparse
import torch
from benchmarks import SplitMNIST
from trainers import NStepKLVCLTrainer
from modules import NStepKLVCL
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    sns.set_style("darkgrid")
    sns.set_context("paper")
    sns.set(font_scale=2)

    # Use seaborn to create the plot
    fig, axs = plt.subplots(1, 5, figsize=(10 * 5,  6 * 1))

    ############################### N-Step / Beta Ablation ###########################################
    betas = [5e-2]#[1e-5, 1e-3, 1e-2, 5e-2, 1e-1, 1.0]
    steps = [1, 2, 3, 4, 5]

    for idx, n_step in enumerate(steps):
        task_plot_dfs = []
        df = pd.read_csv(f'split_mnist_nstep_ablation_final_{n_step}.csv')
        task_plot_dfs.append(df)
    
        plot_values(task_plot_dfs, 5, axs[idx], idx, n_step, f'n-Step = {n_step}')

    
    plt.suptitle('Split MNIST: N-Step TD-VCL Ablation')
    plt.tight_layout()
    plt.savefig('./split_mnist_nsteptdvcl_ablation.png', bbox_inches='tight')
    plt.savefig('./split_mnist_nsteptdvcl_ablation.eps', bbox_inches='tight')
    plt.savefig('./split_mnist_nsteptdvcl_ablation.pdf', bbox_inches='tight')

def generate_plt_dfs(results, method):
    df = pd.DataFrame(results)
    df.columns = df.columns + 1
    df_melted = df.melt(var_name='# tasks', value_name='Accuracy')
    df_melted['Method'] = [method] * len(df_melted)
    return df_melted

def plot_values(dfs, num_tasks, ax, idx, n_step, title):
    df_melted = pd.concat(dfs)

    sns.pointplot(x='# tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax, legend=(idx == 0))

    # Set the y-axis limits and ticks
    ax.set_ylim(0.85, 1.05)
    ax.set_yticks(np.arange(0.85, 1.03, 0.05))

    # Set the x-axis ticks
    ax.set_xticks(np.arange(0, num_tasks, 1))

    if idx == 0:
        ax.legend(title='Method', loc='lower left')
    
    ax.set_title(title)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Permuted MNIST.')
    parser.add_argument('--epochs_per_task', type=int, default=100,
                        help='The number of tasks.')
    parser.add_argument('--layers', type=str, default="[256, 256]",
                        help='The hidden layers.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='The number of tasks.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The learning rate.')
    parser.add_argument('--single_head', type=bool, default=False,
                        help='The learning rate.')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                        help='Train/Valid data ratio.')
    parser.add_argument('--enable_early_stopping', type=bool, default=True,
                        help='Whether to enable early stopping')
    parser.add_argument('--es_patience', type=int, default=5,
                        help='Early Stopping patience.')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='The number of experiment seeds.')
    
    

    args = parser.parse_args()
    main(args)
  