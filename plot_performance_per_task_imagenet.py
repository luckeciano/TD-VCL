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
    ax.set_ylim(lower_ylim, 0.66)
    ax.set_yticks(np.arange(lower_ylim, 0.66, 0.05))

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

tiny_imagenet_online_mle,  tiny_imagenet_online_mle_tasks = load_results("results/tiny_imagenet_online_mle_results.pkl") # This pickle contains all results
tiny_imagenet_batch_mle,  tiny_imagenet_batch_mle_tasks = load_results("results/tiny_imagenet_batch_mle_results.pkl") # This pickle contains all results
tiny_imagenet_vcl,  tiny_imagenet_vcl_tasks = load_results("results/tiny_imagenet_vcl_results.pkl") # This pickle contains all results
tiny_imagenet_vcl_coreset,  tiny_imagenet_vcl_coreset_tasks = load_results("results/tiny_imagenet_vcl_coreset_tasks.pkl") # This pickle contains all results
tiny_imagenet_nstepkl,  tiny_imagenet_nstepkl_tasks = load_results("results/tiny_imagenet_nstepkl_taskids_results.pkl") # This pickle contains all results
tiny_imagenet_tdvcl,  tiny_imagenet_tdvcl_tasks = load_results("results/tiny_imagenet_tdvclids_results.pkl") # This pickle contains all results

dfs = [tiny_imagenet_online_mle_tasks, tiny_imagenet_batch_mle_tasks, tiny_imagenet_vcl_tasks, tiny_imagenet_vcl_coreset_tasks, tiny_imagenet_nstepkl_tasks, tiny_imagenet_tdvcl_tasks]


sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=2.5)
figs, axs = plt.subplots(2, 5, figsize=(6 * 5, 8 * 2))

for i in range(10):
    plot_task_values(axs[i // 5][i % 5], [df[i][0] for df in dfs], i + 1, num_tasks=10, lower_ylim=0.40, legend=(i == 9), skip_ylabel=(i%5 != 0))

plt.suptitle('TinyImageNet-10: Per Task Performance')
figs.tight_layout()
figs.savefig('./tiny_imagenet_tasks.png', bbox_inches='tight')
figs.savefig('./tiny_imagenet_tasks.eps', bbox_inches='tight')
figs.savefig('./tiny_imagenet_tasks.pdf', bbox_inches='tight')


