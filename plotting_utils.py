import numpy as np
import seaborn as sns
import pandas as pd
import pickle


def save_results(dfs, filename='results.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(dfs, f)


# Load the final two resultant objects
def load_results(filename='results.pkl'):
    with open(filename, 'rb') as f:
        dfs = pickle.load(f)
    return dfs

def generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, method, num_tasks):
    results = np.array(seed_results)
    multitask_plot_dfs.append(generate_multitask_plt_dfs(results, method))

    for i in range(num_tasks):
        task_accuracies = []
        for results_per_task in seed_results_per_task:
            task_accuracies.append(results_per_task[i])
        
        singletask_plot_dfs[i].append(generate_single_task_plt_dfs(np.array(task_accuracies), method, i + 1, num_tasks))
    
    return multitask_plot_dfs, singletask_plot_dfs

def generate_multitask_plt_dfs(results, method):
    df = pd.DataFrame(results)
    df.columns = df.columns + 1
    df_melted = df.melt(var_name='# tasks', value_name='Accuracy')
    df_melted['Method'] = [method] * len(df_melted)
    return df_melted

def generate_single_task_plt_dfs(results, method, task_id, num_tasks):
    df = pd.DataFrame(results)
    df.columns = df.columns + task_id
    for task in range(1, num_tasks):
        if task not in df.columns:
            df[task] = np.nan
    df = df.reindex(sorted(df.columns), axis=1)
    df_melted = df.melt(var_name='Tasks', value_name='Accuracy')
    df_melted['Tasks'] = df_melted['Tasks'].astype(str)
    df_melted['Method'] = [method] * len(df_melted)
    return df_melted

    
def plot_values(ax, dfs, num_tasks, lower_ylim, legend, skip_ylabel=False):
    df_melted = pd.concat(dfs,ignore_index=True)
    # Use seaborn to create the plot
    sns.pointplot(x='# tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax, legend=legend)

    if skip_ylabel:
        ax.set_ylabel("")

    if legend:
        ax.legend(title='Method', loc='lower left')

    # Set the y-axis limits and ticks
    ax.set_ylim(lower_ylim, 1.05)
    ax.set_yticks(np.arange(lower_ylim, 1.01, 0.1))

    # Set the x-axis ticks
    ax.set_xticks(np.arange(0, num_tasks, 1))

def plot_task_values(ax, dfs, task_id, num_tasks, lower_ylim, legend, skip_ylabel):
    df_melted = pd.concat(dfs, ignore_index=True)

    # Use seaborn to create the plot
    sns.pointplot(x='Tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax, legend=legend)

    # Set the title
    ax.set_title(f'Task {task_id}')

    # Set the y-axis limits and ticks
    ax.set_ylim(lower_ylim, 1.05)
    ax.set_yticks(np.arange(lower_ylim, 1.05, 0.1))

    if legend:
        ax.legend(title='Method', loc='lower left')

    if skip_ylabel:
        ax.set_ylabel("")

    # Set the x-axis ticks
    ax.set_xticks([str(i) for i in np.arange(1, num_tasks + 1, 1)])
    ax.set_xticklabels(np.arange(1, num_tasks + 1, 1))