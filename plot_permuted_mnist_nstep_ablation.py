import argparse
import torch
from benchmarks import PermutedMNIST
from trainers import NStepKLVCLTrainer
from modules import NStepKLVCL
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
    # Use seaborn to create the plot
    fig, axs = plt.subplots(2, 3, figsize=(10 * 3, 8 * 2))


    ############################### N-Step / Beta Ablation ###########################################
    steps = [1, 2, 3, 5, 8, 10]

    for idx, n_step in enumerate(steps):
        task_plot_dfs = []
        df = pd.read_csv(f'permuted_mnist_nstep_ablation_final_{n_step}.csv')
        task_plot_dfs.append(df)
    
        plot_values(task_plot_dfs, args.num_tasks, axs[idx // 3][idx % 3], idx, n_step, f'n-Step = {n_step}')

    
    plt.suptitle('PermutedMNIST-Hard: N-Step TD-VCL Ablation')
    plt.tight_layout()
    plt.savefig('./test_permuted_mnist_nsteptdvcl_ablation.png', bbox_inches='tight')
    plt.savefig('./test_permuted_mnist_nsteptdvcl_ablation.eps', bbox_inches='tight')
    plt.savefig('./test_permuted_mnist_nsteptdvcl_ablation.pdf', bbox_inches='tight')

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
    ax.set_ylim(0.5, 1.05)
    ax.set_yticks(np.arange(0.5, 1.05, 0.1))

    # Set the x-axis ticks
    ax.set_xticks(np.arange(0, num_tasks, 1))

    if idx == 0:
        ax.legend(loc='lower left', fontsize=23)
    
    if idx % 3 != 0:
        ax.set_ylabel('')

    if idx % 3 != 1:
        ax.set_xlabel('')
    else:
        ax.set_xlabel("Number of Observed Tasks")
    
    ax.set_title(title)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Permuted MNIST.')
    parser.add_argument('--num_tasks', type=int, default=10,
                        help='The number of tasks.')

    args = parser.parse_args()
    main(args)
  