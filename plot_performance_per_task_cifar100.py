import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plotting_utils import load_results

def plot_task_values(ax, dfs, task_id, num_tasks, lower_ylim, legend, skip_ylabel):
    df_melted = pd.concat(dfs, ignore_index=True)
    df_melted.loc[df_melted['Method'] == 'TD(\u03BB)-VCL (MH)', 'Method'] = 'TD(\u03BB)-VCL'

    # Use seaborn to create the plot
    sns.pointplot(x='Tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax, legend=legend)

    # Set the title
    ax.set_title(f'Task {task_id}')

    # Set the y-axis limits and ticks
    ax.set_ylim(lower_ylim, 0.77)
    ax.set_yticks(np.arange(lower_ylim, 0.77, 0.05))

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

cifar100_online_mle,  cifar100_online_mle_tasks = load_results("results/split_cifar100_online_mle_results.pkl") # This pickle contains all results
cifar100_batch_mle,  cifar100_batch_mle_tasks = load_results("results/split_cifar100_batch_mle_results.pkl") # This pickle contains all results
cifar100_batch_mle_tasks = [[df[1]] for df in cifar100_batch_mle_tasks]
cifar100_vcl,  cifar100_vcl_tasks = load_results("results/split_cifar100_vcl_results.pkl") # This pickle contains all results
cifar100_vcl_coreset,  cifar100_vcl_coreset_tasks = load_results("results/split_cifar100_vcl_coreset_results.pkl") # This pickle contains all results
cifar100_nstepkl,  cifar100_nstepkl_tasks = load_results("results/split_cifar100_nstepkl_taskids_results.pkl") # This pickle contains all results
cifar100_tdvcl,  cifar100_tdvcl_tasks = load_results("results/split_cifar100_tdvclids_results.pkl") # This pickle contains all results

dfs = [cifar100_online_mle_tasks, cifar100_batch_mle_tasks, cifar100_vcl_tasks, cifar100_vcl_coreset_tasks, cifar100_nstepkl_tasks, cifar100_tdvcl_tasks]

num_tasks = 3

sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=2.5)
figs, axs = plt.subplots(2, 5, figsize=(6 * 5, 8 * 2))

for i in range(10):
    plot_task_values(axs[i // 5][i % 5], [df[i][0] for df in dfs], i + 1, num_tasks=10, lower_ylim=0.50, legend=(i == 9), skip_ylabel=(i%5 != 0))
    # plot_task_values(axs[i // 5][i % 5], cifar100_batch_mle_tasks[i], i + 1, num_tasks=10, lower_ylim=0.15, legend=(i == 9), skip_ylabel=(i%5 != 0))
    # plot_task_values(axs[i // 5][i % 5], cifar100_vcl_tasks[i], i + 1, num_tasks=10, lower_ylim=0.15, legend=(i == 9), skip_ylabel=(i%5 != 0))
    # plot_task_values(axs[i // 5][i % 5], cifar100_vcl_coreset_tasks[i], i + 1, num_tasks=10, lower_ylim=0.15, legend=(i == 9), skip_ylabel=(i%5 != 0))
    # plot_task_values(axs[i // 5][i % 5], cifar100_nstepkl_tasks[i], i + 1, num_tasks=10, lower_ylim=0.15, legend=(i == 9), skip_ylabel=(i%5 != 0))
    # plot_task_values(axs[i // 5][i % 5], cifar100_tdvcl_tasks[i], i + 1, num_tasks=10, lower_ylim=0.15, legend=(i == 9), skip_ylabel=(i%5 != 0))

plt.suptitle('CIFAR100-10: Per Task Performance')
figs.tight_layout()
figs.savefig('./cifar100_tasks.png', bbox_inches='tight')
figs.savefig('./cifar100_tasks.eps', bbox_inches='tight')
figs.savefig('./cifar100_tasks.pdf', bbox_inches='tight')


