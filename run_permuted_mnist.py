import argparse
import torch
from sklearn.model_selection import train_test_split
from benchmarks import PermutedMNIST
from trainers import StandardClassifierTrainer, VCLTrainer
from modules import MultiHeadMLP, VCL
import numpy as np
import pandas as pd
import random
import utils
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [random.randint(0, 100) for _ in range(args.num_seeds)]

    ############################### Online MLE ###########################################

    # seed_results = []
    # for seed in seeds:
    #     perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
    #     ft_size, num_classes = perm_mnist.get_dims()

        
    #     model = MultiHeadMLP(ft_size, num_classes, args.layers, 'relu', num_classes)
    #     standard_classifier_trainer = StandardClassifierTrainer(model, args, device)

    #     test_accuracies = train_eval_loop(perm_mnist, model, standard_classifier_trainer, seed, seed_results)
    #     seed_results.append(test_accuracies)

    # results = np.array(seed_results)
    # plot_values(results, method='Online MLE')

    ############################### VCL ###########################################
    seed_results = []
    for seed in seeds:
        perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
        ft_size, num_classes = perm_mnist.get_dims()

        # map_model = get_MAP_weights(ft_size, num_classes, device, perm_mnist, seed)
        
        perm_mnist.reset_env()
        model = VCL(ft_size, num_classes, args.layers, 'relu', mle_model=None, n_heads=1)
        vcl_trainer = VCLTrainer(model, args, device, no_kl=False)

        test_accuracies = train_eval_loop(perm_mnist, model, vcl_trainer, seed, seed_results)
        seed_results.append(test_accuracies)

    results = np.array(seed_results)
    plot_values(results, method='VCL')

    plt.tight_layout()
    plt.savefig('./permuted_mnist.png', bbox_inches='tight')

def get_MAP_weights(ft_size, num_classes, device, perm_mnist, seed):
    model = MultiHeadMLP(ft_size, num_classes, args.layers, 'relu', n_heads=1)
    mle_trainer = StandardClassifierTrainer(model, args, device, weight_decay=0.01)
    model.new_task(0, args.single_head)
    train_dataloader, valid_dataloader, test_dataloader = generate_dataloaders(perm_mnist, [], [], seed)
    mle_trainer.train(args.epochs_per_task, train_dataloader, valid_dataloader)

    print(f"Test Accuracy after for MLE model:")
    acc, _ = mle_trainer.evaluate([test_dataloader], single_head=True)
    return model

def train_eval_loop(perm_mnist, model, trainer, seed, seed_results):
    x_test_sets = []
    y_test_sets = []
    test_accuracies = []

    for task_id in range(perm_mnist.max_iter):
        model.new_task(task_id, args.single_head)
        train_dataloader, valid_dataloader, test_dataloader = generate_dataloaders(perm_mnist, x_test_sets, y_test_sets, seed)
        
        trainer.train(args.epochs_per_task, train_dataloader, valid_dataloader)

        print(f"Test Accuracy after task {task_id}:")
        acc, _ = trainer.evaluate([test_dataloader], single_head=True)
        test_accuracies.append(acc)
    
    return test_accuracies



def generate_dataloaders(perm_mnist, x_test_sets, y_test_sets, seed):
    x_train, y_train, x_test, y_test = perm_mnist.next_task()

    x_test_sets.append(x_test)
    y_test_sets.append(y_test)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=args.valid_ratio, random_state=seed)

    train_dataloader = utils.get_dataloader(x_train, y_train, args.batch_size, shuffle=True)
    valid_dataloader = utils.get_dataloader(x_valid, y_valid, args.batch_size, shuffle=True)

    test_dataloader = utils.get_dataloader(np.vstack(x_test_sets), np.vstack(y_test_sets), args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, valid_dataloader, test_dataloader
    


def plot_values(results, method):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    df = pd.DataFrame(results)
    df.columns = df.columns + 1
    df_melted = df.melt(var_name='# tasks', value_name='Accuracy')
    df_melted['Method'] = [method] * len(df_melted)

    # Use seaborn to create the plot
    plt.figure(figsize=(10, 6))
    sns.pointplot(x='# tasks', y='Accuracy', data=df_melted, hue='Method', errorbar='ci', marker='o', capsize=.1)

    # Set the title
    plt.title('Permuted MNIST')

    # Set the y-axis limits and ticks
    plt.ylim(0.3, 1.1)
    plt.yticks(np.arange(0.3, 1.05, 0.1))

    # Set the x-axis ticks
    plt.xticks(np.arange(0, df.shape[1], 1))

    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    

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
    parser.add_argument('--num_seeds', type=int, default=2,
                        help='The number of experiment seeds.')
    
    

    args = parser.parse_args()
    main(args)
  