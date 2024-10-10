import argparse
import torch
from benchmarks import PermutedMNIST
from trainers import OnlineMLETrainer, BatchMLETrainer
from modules import MultiHeadMLP
import numpy as np
import pandas as pd
import random
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_utils import save_results


def main(args):
    sns.set_style("darkgrid")
    sns.set_context("paper")
    sns.set(font_scale=2)
    plot_dfs = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [random.randint(0, 100) for _ in range(args.num_seeds)]
    # Use seaborn to create the plot
    fig, axs = plt.subplots(1, 3, figsize=(10 * 3, 12))

    ############################### Online MLE ###########################################

    seed_results = []
    for seed in seeds:
        perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
        ft_size, num_classes = perm_mnist.get_dims()

        
        model = MultiHeadMLP(ft_size, num_classes, args.layers, 'relu', num_classes)
        online_mle_trainer = OnlineMLETrainer(model, args, device)

        test_accuracies, _ = online_mle_trainer.train_eval_loop(perm_mnist, model, args, seed)
        seed_results.append(test_accuracies)

    results = np.array(seed_results)
    plot_dfs.append(generate_plt_dfs(results, 'Online MLE'))
    save_results(plot_dfs, filename="results/mnist_memory_online_mle.pkl")

    ############################### MLE Past Data ###########################################
    configs = [[(10, 60000), (10, 10000), (10, 1000), (10, 200), (10, 50), (10, 10)],
               [(5, 60000), (5, 10000), (5, 1000), (5, 200), (5, 50), (5, 10)],
               [(2, 60000), (2, 10000), (2, 1000), (2, 200), (2, 50), (2, 10)]]

    for idx, task_config in enumerate(configs):
        task_plot_dfs = []
        for config in task_config:
            num_tasks_mem, task_mem_size = config
            seed_results = []
            for seed in seeds:
                perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
                ft_size, num_classes = perm_mnist.get_dims()
                
                model = MultiHeadMLP(ft_size, num_classes, args.layers, 'relu', num_classes)
                batch_mle_trainer = BatchMLETrainer(model, args, device, num_tasks_mem, task_mem_size)

                test_accuracies, _ = batch_mle_trainer.train_eval_loop(perm_mnist, model, args, seed)
                seed_results.append(test_accuracies)

            results = np.array(seed_results)
            task_plot_dfs.append(generate_plt_dfs(results, f'T = {num_tasks_mem}, B = {task_mem_size}'))
        
        save_results(task_plot_dfs, filename=f"results/mnist_memory_batch_mle_{idx}.pkl")
        plot_values(plot_dfs + task_plot_dfs, args.num_tasks, axs[idx], idx)

    
    plt.suptitle('Permuted MNIST - External Memory Ablation')
    plt.tight_layout()
    plt.savefig('./permuted_mnist_memory.png', bbox_inches='tight')
    plt.savefig('./permuted_mnist_memory.pdf', bbox_inches='tight')
    plt.savefig('./permuted_mnist_memory.eps', bbox_inches='tight')

def get_MAP_weights(ft_size, num_classes, device, perm_mnist, seed):
    model = MultiHeadMLP(ft_size, num_classes, args.layers, 'relu', n_heads=1)
    mle_trainer = OnlineMLETrainer(model, args, device, weight_decay=0.01)
    model.new_task(0, args.single_head)
    train_dataloader, valid_dataloader, test_dataloader = utils.generate_dataloaders(perm_mnist, [], [], args.batch_size, seed)
    mle_trainer.train(args.epochs_per_task, train_dataloader, valid_dataloader)

    print(f"Test Accuracy after for MLE model:")
    acc, _ = mle_trainer.evaluate([test_dataloader], single_head=True)
    return model

def generate_plt_dfs(results, method):
    df = pd.DataFrame(results)
    df.columns = df.columns + 1
    df_melted = df.melt(var_name='# tasks', value_name='Accuracy')
    df_melted['Method'] = [method] * len(df_melted)
    return df_melted

def plot_values(dfs, num_tasks, ax, idx):
    df_melted = pd.concat(dfs)

    sns.pointplot(x='# tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1, ax=ax)

    # Set the y-axis limits and ticks
    ax.set_ylim(0.3, 1.1)
    ax.set_yticks(np.arange(0.3, 1.05, 0.1))

    # Set the x-axis ticks
    ax.set_xticks(np.arange(0, num_tasks, 1))

    if idx != 0:
        ax.set_ylabel("")

    ax.legend(title='Method', loc='lower left')
    

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
  