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
    
    ax.set_xlabel("Number of Observed Tasks")

    if legend:
        leg = ax.legend(title='Method', loc='upper center', fontsize=17, title_fontsize=17, ncol=3, bbox_to_anchor=(0.60, 1.01))

        # Customize the legend text
        bold_labels = ["N-Step TD-VCL", "TD(λ)-VCL"]
        for text in leg.get_texts():
            if text.get_text() in bold_labels:
                text.set_fontweight('bold')

    # Set the y-axis limits and ticks
    ax.set_ylim(lower_ylim, upper_ylim)
    ax.set_yticks(np.arange(lower_ylim, upper_ylim, 0.05))

    # Set the x-axis ticks
    ax.set_xticks(np.arange(0, num_tasks, 1))

    ax.set_title(title)

permuted_mnist, _ = load_results("results/permuted_mnist_tdvcl_results.pkl") # This pickle contains all results
split_mnist, _ = load_results("results/split_mnist_tdvcl_results.pkl") # This pickle contains all results
split_notmnist, _ = load_results("results/split_notmnist_tdvcl_results.pkl") # This pickle contains all results
num_tasks = 3

sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=3.0)

fig = plt.figure(figsize=(13, 10))
ax = fig.gca()
plot_values(ax, permuted_mnist, num_tasks=10, lower_ylim=0.75, upper_ylim=1.01, legend=True, title="PermutedMNIST-Hard")
# plot_values(axs[1], split_mnist, num_tasks=5, lower_ylim=0.5, upper_ylim=1.05, legend=False, title="Split MNIST")
# plot_values(axs[2], split_notmnist, num_tasks=5, lower_ylim=0.45, upper_ylim=0.85, legend=False, title="Split NotMNIST")
fig.tight_layout()
fig.savefig('./test_pickle_permuted_mnist.png', bbox_inches='tight')
fig.savefig('./test_pickle_permuted_mnist.eps', bbox_inches='tight')
fig.savefig('./test_pickle_permuted_mnist.pdf', bbox_inches='tight')


