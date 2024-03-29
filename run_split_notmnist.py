import argparse
import torch
from sklearn.model_selection import train_test_split
from benchmarks import SplitNotMNIST
from trainers import StandardClassifierTrainer
from modules import MultiHeadMLP, MultiHeadMNISTCNN
import numpy as np
import random
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [random.randint(0, 100) for _ in range(args.num_seeds)]
    seed_results = []
    seed_results_per_task = []
    for seed in seeds:
        split_notmnist = SplitNotMNIST()
        ft_size, num_classes = split_notmnist.get_dims()

        # Standard Classifier
        model = MultiHeadMLP(ft_size, num_classes, args.layers, n_heads=5)
        standard_classifier_trainer = StandardClassifierTrainer(model, args, device)

        x_test_sets = []
        y_test_sets = []
        test_accuracies = []
        test_accuracies_per_task = {i: [] for i in range(split_notmnist.max_iter)}

        for task_id in range(split_notmnist.max_iter):
            model.set_task(task_id)
            x_train, y_train, x_valid, y_valid, x_test, y_test = split_notmnist.next_task()

            x_test_sets.append(x_test)
            y_test_sets.append(y_test)

            train_dataloader = utils.get_dataloader(x_train, y_train, args.batch_size, shuffle=True)
            valid_dataloader = utils.get_dataloader(x_valid, y_valid, args.batch_size, shuffle=True)

            test_dataloaders = utils.get_dataloaders(x_test_sets, y_test_sets, args.batch_size, shuffle=False)
            standard_classifier_trainer.train(args.epochs_per_task, train_dataloader, valid_dataloader)

            print(f"Test Accuracy after task {task_id}:")
            acc, acc_tasks = standard_classifier_trainer.evaluate(test_dataloaders, single_head=args.single_head)
            test_accuracies.append(acc)
            for idx, task_acc in enumerate(acc_tasks):
                test_accuracies_per_task[idx].append(task_acc)
        
        seed_results.append(test_accuracies)
        seed_results_per_task.append(test_accuracies_per_task)

    sns.set_style("darkgrid")
    sns.set_context("paper")
    
    fig, axs = plt.subplots(1, 6, figsize=(10 * 6, 6))
    for i in range(5):
        task_accuracies = []
        for results_per_task in seed_results_per_task:
            task_accuracies.append(results_per_task[i])
        
        plot_task_values(axs[i], np.array(task_accuracies), i + 1, 5, 'Online MLE')
    
    plot_values(axs[5], seed_results, 'Online MLE')

    # Dummy plot for centralized legend
    ax_dummy = fig.add_subplot(111, frame_on=False)
    ax_dummy.plot([], [], label='Online MLE')
    ax_dummy.set_xticks([])
    ax_dummy.set_yticks([])

    handles, labels = ax_dummy.get_legend_handles_labels()
    fig.legend(handles, labels, title='Method', bbox_to_anchor=(0.5, 1.02), loc='lower center', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('split_notmnist.png', bbox_inches='tight')

    
def plot_values(ax, results, method):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    df = pd.DataFrame(results)
    df.columns = df.columns + 1
    df_melted = df.melt(var_name='# tasks', value_name='Accuracy')
    df_melted['Method'] = [method] * len(df_melted)

    ax.set_title(f'Average')

    # Use seaborn to create the plot
    sns.pointplot(x='# tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax, legend=False)

    # Set the y-axis limits and ticks
    ax.set_ylim(0.3, 1.05)
    ax.set_yticks(np.arange(0.3, 1.01, 0.1))

    # Set the x-axis ticks
    ax.set_xticks(np.arange(0, df.shape[1], 1))

def plot_task_values(ax, results, task_id, num_tasks, method):
    df = pd.DataFrame(results)
    df.columns = df.columns + task_id
    for task in range(1, num_tasks):
        if task not in df.columns:
            df[task] = np.nan
    df = df.reindex(sorted(df.columns), axis=1)
    df_melted = df.melt(var_name='Tasks', value_name='Accuracy')
    df_melted['Tasks'] = df_melted['Tasks'].astype(str)
    df_melted['Method'] = [method] * len(df_melted)

    # Use seaborn to create the plot
    sns.pointplot(x='Tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax, legend=False)

    # Set the title
    ax.set_title(f'Task {task_id}')

    # Set the y-axis limits and ticks
    ax.set_ylim(0.3, 1.05)
    ax.set_yticks(np.arange(0.3, 1.05, 0.1))

    # Set the x-axis ticks
    ax.set_xticks([str(i) for i in np.arange(1, num_tasks + 1, 1)])
    ax.set_xticklabels(np.arange(1, num_tasks + 1, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Permuted MNIST.')
    parser.add_argument('--epochs_per_task', type=int, default=100,
                        help='The number of epochs.')
    parser.add_argument('--layers', type=str, default="[150, 150, 150, 150]",
                        help='The hidden layers.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='The batch size.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The learning rate.')
    parser.add_argument('--single_head', type=bool, default=False,
                        help='Whether use single or multiple heads.')
    parser.add_argument('--enable_early_stopping', type=bool, default=True,
                        help='Whether to enable early stopping')
    parser.add_argument('--es_patience', type=int, default=5,
                        help='Early Stopping patience.')
    parser.add_argument('--num_seeds', type=int, default=2,
                        help='The number of experiment seeds.')
    

    args = parser.parse_args()
    main(args)
