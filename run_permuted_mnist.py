import argparse
import torch
from benchmarks import PermutedMNIST
from trainers import OnlineMLETrainer, BatchMLETrainer, VCLTrainer, NStepKLVCLTrainer, VCLCoreSetTrainer, TemporalDifferenceVCLTrainer, UCBTrainer, TemporalDifferenceUCBTrainer
from data_structures import  get_random_coreset
from modules import MultiHeadMLP, VCL, NStepKLVCL,TemporalDifferenceVCL, UCL, TemporalDifferenceUCL
import random
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_utils import generate_df_results, plot_task_values, plot_values, save_results


def main(args):
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [random.randint(0, 100) for _ in range(args.num_seeds)]

    # PermutedMNIST Hard Configuration
    num_tasks_mem = 2
    task_mem_size = 200

    multitask_plot_dfs = []
    singletask_plot_dfs = [[], [], [], [], [], [], [], [], [], []]

    # ############################### Online MLE ###########################################

    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
    #     ft_size, num_classes = perm_mnist.get_dims()

        
    #     model = MultiHeadMLP(ft_size, num_classes, args.layers, 'relu', num_classes)
    #     online_mle_trainer = OnlineMLETrainer(model, args, device)

    #     test_accuracies, test_accuracies_per_task = online_mle_trainer.train_eval_loop(perm_mnist, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'Online MLE', args.num_tasks)
    # save_results(multitask_plot_dfs, singletask_plot_dfs, filename="results/permuted_mnist_online_mle_results.pkl")

    # # ############################### Batch MLE ###########################################

    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
    #     ft_size, num_classes = perm_mnist.get_dims()
        
    #     model = MultiHeadMLP(ft_size, num_classes, args.layers, 'relu', num_classes)
    #     batch_mle_trainer = BatchMLETrainer(model, args, device, num_tasks_mem, task_mem_size)

    #     test_accuracies, test_accuracies_per_task = batch_mle_trainer.train_eval_loop(perm_mnist, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'Batch MLE', args.num_tasks)
    # save_results(multitask_plot_dfs, singletask_plot_dfs, filename="results/permuted_mnist_batch_mle_results.pkl")
    
    # # ############################### VCL ###########################################
    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
    #     ft_size, num_classes = perm_mnist.get_dims()
        
    #     model = VCL(ft_size, num_classes, args.layers, 'relu', mle_model=None, n_heads=1, lambd_logvar=-5.0)
    #     vcl_trainer = VCLTrainer(model, args, device, beta=5e-3, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(perm_mnist, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'VCL', args.num_tasks)
    # save_results(multitask_plot_dfs, singletask_plot_dfs, filename="results/permuted_mnist_vcl_results.pkl")

    # # ############################# VCL with Core Set #############################################

    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
    #     ft_size, num_classes = perm_mnist.get_dims()

    #     model = VCL(ft_size, num_classes, args.layers, 'relu', mle_model=None, n_heads=1)
    #     vcl_trainer = VCLCoreSetTrainer(model, args, device, coreset_method=get_random_coreset, K=task_mem_size, max_tasks=num_tasks_mem)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(perm_mnist, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'VCL CoreSet', args.num_tasks)
    # save_results(multitask_plot_dfs, singletask_plot_dfs, filename="results/permuted_mnist_vcl_coreset_results.pkl")

    # # ############################# N-Step KL VCL #############################################

    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
    #     ft_size, num_classes = perm_mnist.get_dims()

    #     n_step = 5
    #     model = NStepKLVCL(ft_size, num_classes, n_step, args.layers, 'relu', n_heads=1)
    #     vcl_trainer = NStepKLVCLTrainer(model, args, device, n_step, num_tasks_mem, task_mem_size, beta=5e-3, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(perm_mnist, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'N-Step TD-VCL', args.num_tasks)
    # save_results(multitask_plot_dfs, singletask_plot_dfs, filename="results/permuted_mnist_nstepkl_results.pkl")

    # ############################# TD(lambda)-VCL #############################################

    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
    #     ft_size, num_classes = perm_mnist.get_dims()

    #     n_step = 8
    #     lambd = 0.5
    #     beta = 1e-3
    #     model = TemporalDifferenceVCL(ft_size, num_classes, n_step, lambd, args.layers, 'relu', n_heads=1)
    #     tdvcl_trainer = TemporalDifferenceVCLTrainer(model, args, device, n_step, lambd, num_tasks_mem, task_mem_size, beta=beta, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = tdvcl_trainer.train_eval_loop(perm_mnist, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)
        
    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'TD(\u03BB)-VCL', num_tasks=args.num_tasks)
    # save_results(multitask_plot_dfs, singletask_plot_dfs, filename="results/permuted_mnist_tdvcl_results.pkl")

    # ############################# UCL #############################################

    # seed_results = []
    # seed_results_per_task = []
    # for seed in seeds:
    #     perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
    #     ft_size, num_classes = perm_mnist.get_dims()
        
    #     model = UCL(ft_size, num_classes, args.layers, 'relu', mle_model=None, n_heads=1, lambd_logvar=-10.0, ratio=0.5, alpha=1.0, beta=0.001, gamma=0.01)
    #     ucl_trainer = VCLTrainer(model, args, device, beta=0.005, no_kl=False) # Same trainer as VCL

    #     test_accuracies, test_accuracies_per_task = ucl_trainer.train_eval_loop(perm_mnist, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)
        
    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'UCL', num_tasks=args.num_tasks)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/permuted_mnist_ucl_results.pkl")


    ############################# TD-UCL #############################################

    # seed_results = []
    # seed_results_per_task = []
    # n_step = 5
    # lambd = 0.99
    # for seed in seeds:
    #     perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
    #     ft_size, num_classes = perm_mnist.get_dims()
        
    #     model = TemporalDifferenceUCL(ft_size, num_classes, n_step, lambd, args.layers, lambda_logvar=-10.0, ratio=0.5, alpha=1.0, beta=0.001, gamma=0.01, activation_fn='relu', n_heads=1)
    #     tducl_trainer = TemporalDifferenceVCLTrainer(model, args, device, n_step, lambd, num_tasks_mem, task_mem_size, beta=0.001, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = tducl_trainer.train_eval_loop(perm_mnist, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)
        
    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'TD-UCL', num_tasks=args.num_tasks)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/permuted_mnist_tducl_results.pkl")

    ############################### UCB ###########################################
    # seed_results = []
    # seed_results_per_task = []
    # prior_patience = args.es_patience
    # args.es_patience = 20 # UCB is more robust to overfitting
    # for seed in seeds:
    #     perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
    #     ft_size, num_classes = perm_mnist.get_dims()
        
    #     model = VCL(ft_size, num_classes, args.layers, 'relu', mle_model=None, n_heads=1, lambd_logvar=-5.0)
    #     vcl_trainer = UCBTrainer(model, args, device, beta=0.01, alpha=1.0, no_kl=False)

    #     test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(perm_mnist, model, args, seed)
    #     seed_results.append(test_accuracies)
    #     seed_results_per_task.append(test_accuracies_per_task)

    # multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'UCB', args.num_tasks)
    # save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/permuted_mnist_ucb_results.pkl")
    # args.es_patience = prior_patience

    ############################### TD-UCB ###########################################
    print("TD-UCB")
    seed_results = []
    seed_results_per_task = []
    prior_patience = args.es_patience
    n_step = 3
    lambd = 0.9
    args.es_patience = 20 # UCB is more robust to overfitting
    for seed in seeds:
        perm_mnist = PermutedMNIST(max_iter=args.num_tasks, seed=seed)
        ft_size, num_classes = perm_mnist.get_dims()
        
        model = TemporalDifferenceVCL(ft_size, num_classes, n_step, lambd, args.layers, 'relu', n_heads=1)
        vcl_trainer = TemporalDifferenceUCBTrainer(model, args, device, n_step, lambd, num_tasks_mem, task_mem_size, alpha=1.0, beta=0.005, no_kl=False)

        test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(perm_mnist, model, args, seed)
        seed_results.append(test_accuracies)
        seed_results_per_task.append(test_accuracies_per_task)

    multitask_plot_dfs, singletask_plot_dfs = generate_df_results(seed_results, seed_results_per_task, multitask_plot_dfs, singletask_plot_dfs, 'TD-UCB', args.num_tasks)
    save_results((multitask_plot_dfs, singletask_plot_dfs), filename="results/permuted_mnist_tducb_results.pkl")
    args.es_patience = prior_patience

    sns.set_style("darkgrid")
    sns.set_context("paper")
    sns.set(font_scale=2)
    
    # Use seaborn to create the plot
    figs, axs = plt.subplots(2, 5, figsize=(10 * 5, 6 * 2))

    for i in range(10):
        plot_task_values(axs[i // 5][i % 5], singletask_plot_dfs[i], i + 1, num_tasks=10, lower_ylim=0.0, legend=(i == 9), skip_ylabel=(i%5 != 0))
    
    plt.suptitle('Permuted MNIST - Per Task Performance')
    figs.tight_layout()
    
    figs.savefig('./permuted_mnist_tasks.png', bbox_inches='tight')
    figs.savefig('./permuted_mnist_tasks.eps', bbox_inches='tight')
    figs.savefig('./permuted_mnist_tasks.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(10, 12))
    ax = fig.gca()
    plot_values(ax, multitask_plot_dfs, num_tasks=10, lower_ylim=0.3, legend=True)
    plt.suptitle('Permuted MNIST')
    fig.tight_layout()
    fig.savefig('./permuted_mnist.png', bbox_inches='tight')
    fig.savefig('./permuted_mnist.eps', bbox_inches='tight')
    fig.savefig('./permuted_mnist.pdf', bbox_inches='tight')

def get_MAP_weights(ft_size, num_classes, device, perm_mnist, seed):
    model = MultiHeadMLP(ft_size, num_classes, args.layers, 'relu', n_heads=1)
    mle_trainer = OnlineMLETrainer(model, args, device, weight_decay=0.00)
    x_test_sets = []
    y_test_sets = []
    model.new_task(0, args.single_head)
    x_train, y_train, x_valid, y_valid, _ = utils.generate_data_splits(perm_mnist, x_test_sets, y_test_sets, seed, args.valid_ratio)
    train_dataloader, valid_dataloader, test_dataloader = utils.generate_dataloaders(x_train, y_train, x_valid, y_valid, x_test_sets, y_test_sets, args.batch_size, seed)

    mle_trainer.train(args.epochs_per_task, train_dataloader, valid_dataloader)

    print(f"Test Accuracy after for MLE model:")
    acc, _ = mle_trainer.evaluate([test_dataloader], single_head=True)
    return model    

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
  