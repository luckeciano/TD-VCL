import argparse
import torch
from benchmarks import SplitCIFAR10100
from trainers import OnlineMLETrainer, BatchMLETrainer, VCLTrainer, VCLCoreSetTrainer, NStepKLVCLTrainer, TemporalDifferenceVCLTrainer
from data_structures import get_random_coreset
from modules import ConvNet, VCLBayesianConvNet, VCL, NStepKLVCL, TemporalDifferenceVCL
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from plotting_utils import generate_df_results, plot_task_values, plot_values, save_results


def main(args):
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [random.randint(0, 100) for _ in range(args.num_seeds)]
    sns.set_style("darkgrid")
    sns.set_context("paper")
    sns.set(font_scale=2)
    fig, axs = plt.subplots(2, 3, figsize=(10 * 3, 6 * 2))

    # SplitCIFAR Hard Configuration
    num_tasks_mem, task_mem_size = 2, 200

    multitask_plot_dfs = []
    singletask_plot_dfs = [[], [], [], [], []]


    ##################################### Online MLE #################################################
    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100()
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = ConvNet(ft_size, num_heads=10, num_classes=num_classes)
    #     online_mle_trainer = OnlineMLETrainer(model, args, device)

    #     test_accuracies, test_accuracies_per_task = online_mle_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'Online MLE', num_tasks=5)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_online_mle_results.pkl")

    # ##################################### Batch MLE #################################################
    # seed_results = []
    # seed_results_per_task = []
    
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100()
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = ConvNet(ft_size, num_heads=10, num_classes=num_classes)
    #     batch_mle_trainer = BatchMLETrainer(model, args, device, num_tasks_mem, task_mem_size)

    #     test_accuracies, test_accuracies_per_task = batch_mle_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'Batch MLE', num_tasks=5)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_batch_mle_results.pkl")

    ############################### VCL ###########################################
    seed_results = []
    seed_results_per_task = []
    for seed in seeds:
        split_cifar_10100 = SplitCIFAR10100()
        ft_size, num_classes = split_cifar_10100.get_dims()

        model = VCLBayesianConvNet(ft_size, num_heads=11, num_classes=num_classes, lambda_logvar=-12.0)
        vcl_trainer = VCLTrainer(model, args, device, beta=3e-5, no_kl=False)

        test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_10100, model, args, seed)
        seed_results.append(test_accuracies)
        seed_results_per_task.append(test_accuracies_per_task)

    multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'VCL', num_tasks=5)
    save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar10100_vcl_results.pkl")
    
    ############################# VCL with Core Set #####################################

    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR10100()
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = VCLBayesianConvNet(ft_size, num_heads=10, num_classes=num_classes, lambda_logvar=-12.0)
    #     vcl_trainer = VCLCoreSetTrainer(model, args, device, beta=3e-5, coreset_method=get_random_coreset, K=task_mem_size, max_tasks=num_tasks_mem)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'VCL CoreSet', num_tasks=5)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_vcl_coreset_results.pkl")
    
    ############################# N-Step KL VCL #############################################

    seed_results = []
    for seed in seeds:
        split_not_mnist = SplitNotMNIST()
        ft_size, num_classes = split_not_mnist.get_dims()

        n_step = 2
        beta = 0.1
        model = NStepKLVCL(ft_size, num_classes, n_step, args.layers, 'relu', n_heads=1)
        vcl_trainer = NStepKLVCLTrainer(model, args, device, n_step, num_tasks_mem, task_mem_size, beta=beta, no_kl=False)

        test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_not_mnist, model, args, seed)
        seed_results.append(test_accuracies)
        seed_results_per_task.append(test_accuracies_per_task)
        
    multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'N-Step TD-VCL', num_tasks=5)
    save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_notmnist_nstepkl_results.pkl")

    ############################# TD(lambda)-VCL #############################################

    seed_results = []
    for seed in seeds:
        split_not_mnist = SplitNotMNIST()
        ft_size, num_classes = split_not_mnist.get_dims()

        n_step = 3
        lambd = 0.1
        beta = 5e-2
        model = TemporalDifferenceVCL(ft_size, num_classes, n_step, lambd, args.layers, 'relu', n_heads=1)
        tdvcl_trainer = TemporalDifferenceVCLTrainer(model, args, device, n_step, lambd, num_tasks_mem, task_mem_size, beta=beta, no_kl=False)

        test_accuracies, test_accuracies_per_task = tdvcl_trainer.train_eval_loop(split_not_mnist, model, args, seed)
        seed_results.append(test_accuracies)
        seed_results_per_task.append(test_accuracies_per_task)
        
    multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'TD(\u03BB)-VCL', num_tasks=5)
    save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_notmnist_tdvcl_results.pkl")

    for i in range(5):
        plot_task_values(axs[i // 3][i % 3], singletask_plot_dfs[i], i + 1, 5, 0.4, i == 4, i % 3 != 0)
    plot_values(axs[1][2], multitask_plot_dfs, 5, lower_ylim=0.4, legend=False, skip_ylabel=True)
    axs[1][2].set_title(f'Average')

    # # Dummy plot for centralized legend
    # ax_dummy = fig.add_subplot(111, frame_on=False)
    # ax_dummy.plot([], [], label='Online MLE')
    # ax_dummy.set_xticks([])
    # ax_dummy.set_yticks([])

    # handles, labels = ax_dummy.get_legend_handles_labels()
    # fig.legend(handles, labels, title='Method', bbox_to_anchor=(0.5, 1.02), loc='lower center', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('split_not_mnist.png', bbox_inches='tight')
    plt.savefig('split_not_mnist.eps', bbox_inches='tight')
    plt.savefig('split_not_mnist.pdf', bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Permuted MNIST.')
    parser.add_argument('--epochs_per_task', type=int, default=100,
                        help='The number of epochs.')
    parser.add_argument('--layers', type=str, default="[256, 256]",
                        help='The hidden layers.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='The batch size.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The learning rate.')
    parser.add_argument('--single_head', type=bool, default=False,
                        help='Whether use single or multiple heads.')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                        help='Train/Valid data ratio.')
    parser.add_argument('--enable_early_stopping', type=bool, default=True,
                        help='Whether to enable early stopping')
    parser.add_argument('--es_patience', type=int, default=10,
                        help='Early Stopping patience.')
    parser.add_argument('--num_seeds', type=int, default=1,
                        help='The number of experiment seeds.')
    

    args = parser.parse_args()
    main(args)
