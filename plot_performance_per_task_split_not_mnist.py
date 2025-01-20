import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plotting_utils import load_results

def plot_values(ax, dfs, num_tasks, lower_ylim, upper_ylim, legend, title, skip_ylabel=False):
    df_melted = pd.concat(dfs,ignore_index=True)
    # Use seaborn to create the plot
    s = sns.pointplot(x='# tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax, legend=legend)

    # Define the custom order for 'Method'
    method_order = ["Online MLE", "Batch MLE", "VCL", "VCL CoreSet", "N-Step TD-VCL", "TD(λ)-VCL"]

    # Convert the 'Method' column to a categorical type with the specified order, keeping the rest as is
    df_melted['Method'] = pd.Categorical(df_melted['Method'], categories=method_order, ordered=True)

    # Group by '# tasks' and 'Method', calculate mean and 2-sigma error, both rounded to 2 decimal places
    df_summary = df_melted.groupby(['# tasks', 'Method']).agg(
        Average_Accuracy=('Accuracy', lambda x: round(x.mean(), 2)),
        Two_Sigma_Error=('Accuracy', lambda x: round(2 * x.std(), 2))
    ).reset_index()

    # Sort by '# tasks' and then by 'Method' according to the specified order
    df_summary = df_summary.sort_values(by=['# tasks', 'Method']).reset_index(drop=True)

    print(df_summary)

    if skip_ylabel:
        ax.set_ylabel("")
    
    ax.set_xlabel("")

    if legend:
        ax.legend(title='Method', loc='lower left', fontsize=23, title_fontsize=23)

    # Set the y-axis limits and ticks
    ax.set_ylim(lower_ylim, upper_ylim)
    ax.set_yticks(np.arange(lower_ylim + 0.05, upper_ylim, 0.1))

    # Set the x-axis ticks
    ax.set_xticks(np.arange(0, num_tasks, 1))

    ax.set_title(title)

def plot_task_values(ax, dfs, task_id, num_tasks, lower_ylim, upper_ylim, legend, skip_ylabel, task_id_dict):
    df_melted = pd.concat(dfs, ignore_index=True)

    # Use seaborn to create the plot
    sns.pointplot(x='Tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax, legend=legend)

    # Set the title
    ax.set_title(f'Task {task_id_dict[task_id]}')

    # Set the y-axis limits and ticks
    ax.set_ylim(lower_ylim, upper_ylim)
    ax.set_yticks(np.arange(lower_ylim + 0.1, upper_ylim, 0.1))

    if legend:
        ax.legend(title='Method', loc='lower left', fontsize=20, title_fontsize=20)

    if skip_ylabel:
        ax.set_ylabel("")

    if task_id != 2 and task_id != 5:
        ax.set_xlabel('')
    else: ax.set_xlabel('Number of Observed Tasks')

    # Set the x-axis ticks
    ax.set_xticks([str(i) for i in np.arange(1, num_tasks + 1, 1)])
    ax.set_xticklabels(np.arange(1, num_tasks + 1, 1))

_,  permuted_mnist_tasks = load_results("results/permuted_mnist_tdvcl_results.pkl") # This pickle contains all results
# split_mnist, split_mnist_tasks = load_results("results/split_mnist_tdvcl_results.pkl") # This pickle contains all results
# split_mnist, split_mnist_tasks = load_results("results/split_mnist_multihead_tdvcl_results.pkl") # This pickle contains all results
split_notmnist, split_notmnist_tasks = load_results("results/split_notmnist_tdvcl_results.pkl") # This pickle contains all results
num_tasks = 3

task_id_dict = {1: "'A/F'", 2: "'B/G'", 3: "'C/H'", 4: "'D/I'", 5: "'E/J'"}

sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=2.5)
figs, axs = plt.subplots(2, 3, figsize=(8 * 3, 6 * 2))

for i in range(5):
    plot_task_values(axs[i // 3][i % 3], split_notmnist_tasks[i], i + 1, 5, 0.3, 1.02, i == 4, i % 3 != 0, task_id_dict)
plot_values(axs[1][2], split_notmnist, num_tasks=5, lower_ylim=0.45, upper_ylim=0.83, legend=False, title="Permuted MNIST", skip_ylabel=True)
axs[1][2].set_title(f'Average')

plt.suptitle('SplitNotMNIST-Hard: Per Task Performance')

figs.tight_layout()
figs.savefig('./iclr_split_notmnist_tasks.png', bbox_inches='tight')
figs.savefig('./iclr_split_notmnist_tasks.eps', bbox_inches='tight')
figs.savefig('./iclr_split_notmnist_tasks.pdf', bbox_inches='tight')


