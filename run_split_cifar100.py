import argparse
import torch
from benchmarks import SplitCIFAR100
from trainers import OnlineMLETrainer, BatchMLETrainer, VCLTrainer, VCLCoreSetTrainer, MultiHeadVCLCoreSetTrainer, \
    NStepKLVCLTrainer, MultiHeadNStepKLVCLTrainer, TemporalDifferenceVCLTrainer, MultiHeadTDVCLTrainer, UCBTrainer, TemporalDifferenceUCBTrainer, MultiHeadTDUCBTrainer
from data_structures import get_random_coreset
from modules import AlexNet, AlexNetV2, ConvNet, VCLBayesianConvNet, VCLBayesianAlexNet, VCLBayesianAlexNetV2, \
    NStepKLVCLBayesianAlexNet, NStepKLVCLBayesianAlexNetV2, MultiHeadNStepKLVCLBayesianAlexNet, \
    MultiHeadVCLBayesianAlexNet, \
    TDVCLBayesianAlexNet, TDVCLBayesianAlexNetV2, MultiHeadTDVCLBayesianAlexNet, \
    VCL, NStepKLVCL, TemporalDifferenceVCL, \
    UCLBayesianAlexNet, TDUCLBayesianAlexNet, MultiHeadTDUCLBayesianAlexNet
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
    num_tasks_mem, task_mem_size = 0, 0

    multitask_plot_dfs = []
    singletask_plot_dfs = [[], [], [], [], [], [], [], [], [], []]


    ##################################### Online MLE #################################################
    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100()
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = AlexNet(ft_size, num_heads=10, num_classes=num_classes)
    #     online_mle_trainer = OnlineMLETrainer(model, args, device)

    #     test_accuracies, test_accuracies_per_task = online_mle_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'Online MLE', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_online_mle_results.pkl")

    # # # ##################################### Batch MLE #################################################
    # seed_results = []
    # seed_results_per_task = []
    
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100()
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = AlexNet(ft_size, num_heads=10, num_classes=num_classes)
    #     batch_mle_trainer = BatchMLETrainer(model, args, device, num_tasks_mem, task_mem_size)

    #     test_accuracies, test_accuracies_per_task = batch_mle_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'Batch MLE', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_batch_mle_results.pkl")

    # # ############################### VCL ###########################################
    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100()
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = VCLBayesianAlexNet(ft_size, num_heads=10, num_classes=num_classes, lambda_logvar=-12.0, lambda_logvar_mlp=-8.0)
    #     vcl_trainer = VCLTrainer(model, args, device, beta=5e-4, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'VCL', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_vcl_results.pkl")
    
    # ############################# VCL with Core Set #####################################
    # # Uses Task id!
    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100(task_id=True)
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = MultiHeadVCLBayesianAlexNet(ft_size, num_heads=10, num_classes=num_classes, lambda_logvar=-12.0, lambda_logvar_mlp=-8.0)
    #     vcl_trainer = MultiHeadVCLCoreSetTrainer(model, args, device, beta=5e-4, coreset_method=get_random_coreset, K=200, max_tasks=10)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'VCL CoreSet', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_vcl_coreset_results.pkl")
    
    ############################# N-Step KL VCL (with Task Ids) #############################################

    # seed_results = []
    # seed_results_per_task = []
    # use_task_ids = True
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100(task_id=use_task_ids)
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     n_step = 5
    #     beta = 3e-5
    #     model = MultiHeadNStepKLVCLBayesianAlexNet(ft_size, n_step, num_heads=10, num_classes=num_classes, lambda_logvar=-12.0, lambda_logvar_mlp=-8.0)
    #     vcl_trainer = MultiHeadNStepKLVCLTrainer(model, args, device, n_step, num_mem_tasks=10, task_mem_size=200, beta=beta, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)
        
    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'N-Step TD-VCL', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_nstepkl_taskids_results.pkl")

    ############################# N-Step KL VCL (without Task Ids) #############################################
    seed_results = []
    seed_results_per_task = []
    use_task_ids = False
    for seed in seeds:
        split_cifar_100 = SplitCIFAR100(task_id=use_task_ids)
        ft_size, num_classes = split_cifar_100.get_dims()

        n_step = 8
        beta = 1e-4
        model = NStepKLVCLBayesianAlexNet(ft_size, n_step, num_heads=10, num_classes=num_classes, lambda_logvar=-12.0, lambda_logvar_mlp=-8.0)
        vcl_trainer = NStepKLVCLTrainer(model, args, device, n_step, num_mem_tasks=0, task_mem_size=0, beta=beta, no_kl=False)

        test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
        seed_results.append(test_accuracies)
        seed_results_per_task.append(test_accuracies_per_task)
        
    multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'N-Step TD-VCL', num_tasks=10)
    save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_nstepkl_results.pkl")

    ############################# TD(lambda)-VCL (without Task Ids) #############################################
    # print("TDVCL Without ids")

    # seed_results = []
    # seed_results_per_task = []
    # use_task_ids = False
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100(task_id=use_task_ids)
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     n_step = 8
    #     lambd = 0.9
    #     beta = 0.0001
    #     model = TDVCLBayesianAlexNet(ft_size, n_step, lambd, num_classes=num_classes, num_heads=10, lambda_logvar=-12.0, lambda_logvar_mlp=-8.0)
    #     tdvcl_trainer = TemporalDifferenceVCLTrainer(model, args, device, n_step, lambd, num_mem_tasks=0, task_mem_size=0, beta=beta, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = tdvcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)
        
    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'TD(\u03BB)-VCL', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_tdvcl_results.pkl")

    # ############################# TD(lambda)-VCL (with Task Ids) #############################################
    # print("TDVCL With ids")
    # seed_results = []
    # seed_results_per_task = []
    # use_task_ids = True
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100(task_id=use_task_ids)
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     n_step = 10
    #     lambd = 0.5
    #     beta = 1e-5
    #     model = MultiHeadTDVCLBayesianAlexNet(ft_size, n_step, lambd, num_classes=num_classes, num_heads=10, lambda_logvar=-12.0, lambda_logvar_mlp=-8.0)
    #     tdvcl_trainer = MultiHeadTDVCLTrainer(model, args, device, n_step, lambd, num_mem_tasks=10, task_mem_size=200, beta=beta, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = tdvcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)
        
    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'TD(\u03BB)-VCL (MH)', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_tdvclids_results.pkl")

    ############################### UCL ###########################################
    # print("UCL")
    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100()
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = UCLBayesianAlexNet(ft_size, num_heads=10, num_classes=num_classes, lambda_logvar=-20.0, lambda_logvar_mlp=-16.0, ratio=0.5, alpha=1.0, beta=0.001, gamma=0.005)
    #     vcl_trainer = VCLTrainer(model, args, device, beta=1e-4, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'UCL', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_ucl_results.pkl")

    # # ############################### TD-UCL (without Task Ids) ###########################################
    # print("TD-UCL")
    # seed_results = []
    # seed_results_per_task = []
    # n_step = 1
    # lambd = 0.99
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100()
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = TDUCLBayesianAlexNet(ft_size, n_step, lambd, num_heads=10, num_classes=num_classes, lambda_logvar=-20.0, lambda_logvar_mlp=-16.0, ratio=0.5, alpha=1.0, beta=0.001, gamma=0.005)
    #     vcl_trainer = TemporalDifferenceVCLTrainer(model, args, device, n_step, lambd, num_tasks_mem, task_mem_size, beta=1e-4, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'UCL', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_tducl_results.pkl")

    # # ############################### TD-UCL (with Task Ids) ###########################################
    # print("TD-UCL with Task Ids")
    # seed_results = []
    # seed_results_per_task = []
    # n_step = 5
    # lambd = 0.8
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100(task_id=True)
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = MultiHeadTDUCLBayesianAlexNet(ft_size, n_step, lambd, num_heads=10, num_classes=num_classes, lambda_logvar=-20.0, lambda_logvar_mlp=-16.0, ratio=0.5, alpha=1.0, beta=0.001, gamma=0.005)
    #     vcl_trainer = MultiHeadTDVCLTrainer(model, args, device, n_step, lambd, num_mem_tasks=10, task_mem_size=200, beta=1e-5, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'TD-UCL (TIL)', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_tduclids_results.pkl")

    # ############################### UCB ###########################################
    # print("UCB")
    # seed_results = []
    # seed_results_per_task = []
    # prior_patience = args.es_patience
    # args.es_patience = 20
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100()
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = VCLBayesianAlexNet(ft_size, num_heads=10, num_classes=num_classes, lambda_logvar=-10.0, lambda_logvar_mlp=-5.0)
    #     vcl_trainer = UCBTrainer(model, args, device, beta=5e-5, alpha=10.0, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'UCB', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_ucb_results.pkl")
    # args.es_patience = prior_patience

    ############################### TD-UCB (without Task Ids) ###########################################
    # print("TD-UCB")
    # n_step = 2
    # lambd = 0.5
    # seed_results = []
    # seed_results_per_task = []
    # prior_patience = args.es_patience
    # args.es_patience = 20
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100()
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = TDVCLBayesianAlexNet(ft_size, n_step, lambd, num_heads=10, num_classes=num_classes, lambda_logvar=-10.0, lambda_logvar_mlp=-5.0)
    #     vcl_trainer = TemporalDifferenceUCBTrainer(model, args, device, n_step, lambd, num_mem_tasks=0, task_mem_size=0, beta=0.001, alpha=10.0, no_kl=False)
        
    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'TD-UCB', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_tducb_results.pkl")
    # args.es_patience = prior_patience

    # ############################### TD-UCB (with Task Ids) ###########################################
    # print("TD-UCB (with Task Ids)")
    # n_step = 8
    # lambd = 0.8
    # seed_results = []
    # seed_results_per_task = []
    # prior_patience = args.es_patience
    # args.es_patience = 20
    # for seed in seeds:
    #     split_cifar_100 = SplitCIFAR100(task_id=True)
    #     ft_size, num_classes = split_cifar_100.get_dims()

    #     model = MultiHeadTDVCLBayesianAlexNet(ft_size, n_step, lambd, num_heads=10, num_classes=num_classes, lambda_logvar=-10.0, lambda_logvar_mlp=-5.0)
    #     vcl_trainer = MultiHeadTDUCBTrainer(model, args, device, n_step, lambd, num_mem_tasks=10, task_mem_size=200, beta=1e-5, alpha=10.0, no_kl=False)
        
    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'TD-UCB', num_tasks=10)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/split_cifar100_tducbids_results.pkl")
    # args.es_patience = prior_patience



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
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='The number of experiment seeds.')
    

    args = parser.parse_args()
    main(args)
