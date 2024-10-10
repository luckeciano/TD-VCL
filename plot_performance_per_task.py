import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plotting_utils import load_results

def plot_task_values(ax, dfs, task_id, num_tasks, lower_ylim, legend, skip_ylabel):
    df_melted = pd.concat(dfs, ignore_index=True)

    # Use seaborn to create the plot
    sns.pointplot(x='Tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax, legend=legend)

    # Set the title
    ax.set_title(f'Task {task_id}')

    # Set the y-axis limits and ticks
    ax.set_ylim(lower_ylim, 1.05)
    ax.set_yticks(np.arange(lower_ylim + 0.05, 1.05, 0.2))

    if legend:
        ax.legend(title='Method', loc='lower left', fontsize=23)

    if skip_ylabel:
        ax.set_ylabel("")

    if task_id != 3 and task_id != 8:
        ax.set_xlabel('')
    else: ax.set_xlabel('Number of Observed Tasks')

    # Set the x-axis ticks
    ax.set_xticks([str(i) for i in np.arange(1, num_tasks + 1, 1)])
    ax.set_xticklabels(np.arange(1, num_tasks + 1, 1))

_,  permuted_mnist_tasks = load_results("results/permuted_mnist_tdvcl_results.pkl") # This pickle contains all results
split_mnist, _ = load_results("results/split_mnist_tdvcl_results.pkl") # This pickle contains all results
split_notmnist, _ = load_results("results/split_notmnist_tdvcl_results.pkl") # This pickle contains all results
num_tasks = 3

sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=2.5)
figs, axs = plt.subplots(2, 5, figsize=(6 * 5, 8 * 2))

for i in range(10):
    plot_task_values(axs[i // 5][i % 5], split_mnist[i], i + 1, num_tasks=10, lower_ylim=0.15, legend=(i == 9), skip_ylabel=(i%5 != 0))

plt.suptitle('Split MNIST: Per Task Performance')
figs.tight_layout()
figs.savefig('./iclr_permuted_mnist_tasks.png', bbox_inches='tight')
figs.savefig('./iclr_permuted_mnist_tasks.eps', bbox_inches='tight')
figs.savefig('./iclr_permuted_mnist_tasks.pdf', bbox_inches='tight')


