import argparse
import torch
from benchmarks import PermutedMNIST
from trainers import TemporalDifferenceVCLTrainer
from modules import TemporalDifferenceVCL
import numpy as np
import pandas as pd
import random
import utils
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    sns.set_style("darkgrid")
    sns.set_context("paper")
    sns.set(font_scale=3.0)

    ############################### N-Step / Beta Ablation ###########################################
    n_steps = [1, 2, 3, 5, 8, 10]
    lambds = [0.0, 0.1, 0.5, 0.8, 0.9, 0.99]
    betas = [1e-3]

    # beta = 5e-3
    for beta in betas:
        # Use seaborn to create the plot
        fig, axs = plt.subplots(2, 3, figsize=(10 * 3, 8 * 2))

        for idx, n_step in enumerate(n_steps):
            task_plot_dfs = []
            df = pd.read_csv(f'permuted_mnist_nstep_ablation_final_{n_step}_{beta}.csv')
            task_plot_dfs.append(df)
            
            plot_values(task_plot_dfs, args.num_tasks, axs[idx // 3][idx % 3], idx, n_step, beta, f'n-Step: {n_step}')

    
        plt.suptitle('PermutedMNIST-Hard: TD(\u03BB)-VCL Ablation')
        plt.tight_layout()
        plt.savefig(f'./test_permuted_mnist_td_ablation_final_{beta}.png', bbox_inches='tight')
        plt.savefig(f'./test_permuted_mnist_td_ablation_final_{beta}.eps', bbox_inches='tight')
        plt.savefig(f'./test_permuted_mnist_td_ablation_final_{beta}.pdf', bbox_inches='tight')

def generate_plt_dfs(results, method):
    df = pd.DataFrame(results)
    df.columns = df.columns + 1
    df_melted = df.melt(var_name='# tasks', value_name='Accuracy')
    df_melted['Method'] = [method] * len(df_melted)
    return df_melted

def plot_values(dfs, num_tasks, ax, idx, n_step, beta, title):
    df_melted = pd.concat(dfs)

    sns.pointplot(x='# tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax, legend=(idx == 0))

    # Set the y-axis limits and ticks
    ax.set_ylim(0.75, 1.01)
    ax.set_yticks(np.arange(0.75, 1.01, 0.05))

    # Set the x-axis ticks
    ax.set_xticks(np.arange(0, num_tasks, 1))

    if idx == 0:
        ax.legend(loc='lower left', fontsize=23)
    
    if idx % 3 != 0:
        ax.set_ylabel('')

    if idx % 3 != 1:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Number of Observed Tasks')
    
    ax.set_title(title)
    

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
  