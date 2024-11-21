import argparse
import torch
from benchmarks import SplitCIFAR100
from trainers import OnlineMLETrainer, BatchMLETrainer, VCLTrainer, VCLCoreSetTrainer, MultiHeadVCLCoreSetTrainer, \
    NStepKLVCLTrainer, MultiHeadNStepKLVCLTrainer, TemporalDifferenceVCLTrainer, MultiHeadTDVCLTrainer
from data_structures import get_random_coreset
from modules import UCLBayesianAlexNet
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from plotting_utils import generate_df_results, plot_task_values, plot_values, save_results
from heapq import nlargest
import itertools


def main(args):
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [random.randint(0, 100) for _ in range(args.num_seeds)]


    ############################### UCL ###########################################
    alphas = [1.0, 10.0, 0.1, 0.5, 0.01, 0.001]
    betas = [0.001, 0.005, 0.01, 0.1, 1.0]
    gammas = [0.001, 0.005, 0.01, 0.1, 1.0]
    kl_betas = [5e-3, 1e-4, 1e-3, 1e-2, 1e-5]
    lambds_cnn = [-10.0, -8.0, -12.0, -5.0, -16.0, -20.0]
    lambds_mlp = [-10.0, -8.0, -12.0, -5.0, -16.0, -20.0]
    
    combinations = list(itertools.product(alphas, betas, gammas, kl_betas, lambds_cnn, lambds_mlp)) # Shuffle the combinations to ensure random sampling random.shuffle(combinations)
    best_results = []
    
    for alpha, beta, gamma, kl_beta, lambd_cnn, lambd_mlp in combinations:
        print(f"Config: alpha: {alpha}, beta: {beta}, gamma: {gamma}, kl_beta: {kl_beta}, lambd_cnn: {lambd_cnn}, lambd_mlp: {lambd_mlp}")
        seed_results = []
        seed_results_per_task = []
        
        for seed in seeds:
            split_cifar_100 = SplitCIFAR100()
            ft_size, num_classes = split_cifar_100.get_dims()

            model = UCLBayesianAlexNet(ft_size, num_heads=10, num_classes=num_classes, lambda_logvar=lambd_cnn, lambda_logvar_mlp=lambd_mlp, alpha=alpha, beta=beta, gamma=gamma)
            
            vcl_trainer = VCLTrainer(model, args, device, beta=kl_beta, no_kl=False)

            test_accuracies, test_accuracies_per_task = vcl_trainer.train_eval_loop(split_cifar_100, model, args, seed, break_search=True, break_search_min=0.35)
            if test_accuracies is None:
                print("Skipping config")
                break
            seed_results.append(test_accuracies)
            seed_results_per_task.append(test_accuracies_per_task)
            if test_accuracies[-1] < 0.55:
                    print("Skipping config")
                    break

        if seed_results:
            last_term_results = [result[-1] for result in seed_results] 
            avg_test = np.mean(last_term_results)

            config = (beta, lambd_cnn, lambd_mlp)
            best_results.append((avg_test, config))
            best_results = nlargest(10, best_results, key=lambda x: x[0])

            print("Top 10 Configurations:")
            for rank, (avg, params) in enumerate(best_results, 1):
                print(f"Rank {rank}: Avg: {avg}, Params: {params}")

    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Permuted MNIST.')
    parser.add_argument('--epochs_per_task', type=int, default=100,
                        help='The number of epochs.')
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
    parser.add_argument('--num_seeds', type=int, default=3,
                        help='The number of experiment seeds.')
    

    args = parser.parse_args()
    main(args)


                

