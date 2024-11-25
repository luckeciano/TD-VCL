import argparse
import torch
from benchmarks import SplitMNIST
from trainers import UCBTrainer
from modules import VCL
import numpy as np
import random
from heapq import nlargest
import itertools


def main(args):
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [random.randint(0, 100) for _ in range(args.num_seeds)]


    ############################### UCB ###########################################
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    kl_betas = [5e-3, 1e-4, 1e-3, 1e-2, 1e-5]
    lambd_logvars = [-3.0, -5.0, -8.0, -10.0]
    
    
    
    combinations = list(itertools.product(alphas, kl_betas, lambd_logvars)) # Shuffle the combinations to ensure random sampling 
    random.shuffle(combinations)
    best_results = []
    
    for alpha, kl_beta, lambd_logvar in combinations:
        print(f"Config: alpha: {alpha}, kl_beta: {kl_beta}, lambd_logvar: {lambd_logvar}")
        seed_results = []
        seed_results_per_task = []
        for seed in seeds:
            split_mnist = SplitMNIST()
            ft_size, num_classes = split_mnist.get_dims()
            
            model = VCL(ft_size, num_classes, args.layers, 'relu', mle_model=None, n_heads=1, lambd_logvar=lambd_logvar)
            ucb_trainer = UCBTrainer(model, args, device, beta=kl_beta, alpha=alpha, no_kl=False)

            test_accuracies, test_accuracies_per_task = ucb_trainer.train_eval_loop(split_mnist, model, args, seed, break_search=False)
            if test_accuracies is None:
                print("Skipping config")
                break
            seed_results.append(test_accuracies)
            seed_results_per_task.append(test_accuracies_per_task)
            if test_accuracies[-1] < 0.50:
                    print("Skipping config")
                    break

        if seed_results:
            last_term_results = [result[-1] for result in seed_results] 
            avg_test = np.mean(last_term_results)

            config = (alpha, kl_beta, lambd_logvar)
            best_results.append((avg_test, config))
            best_results = nlargest(10, best_results, key=lambda x: x[0])

            print("Top 10 Configurations:")
            for rank, (avg, params) in enumerate(best_results, 1):
                print(f"Rank {rank}: Avg: {avg}, Params: {params}")

    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Split MNIST.')
    parser.add_argument('--epochs_per_task', type=int, default=100,
                        help='The number of tasks.')
    parser.add_argument('--layers', type=str, default="[256, 256]",
                        help='The hidden layers.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='The number of tasks.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The learning rate.')
    parser.add_argument('--single_head', type=bool, default=True,
                        help='The learning rate.')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                        help='Train/Valid data ratio.')
    parser.add_argument('--enable_early_stopping', type=bool, default=True,
                        help='Whether to enable early stopping')
    parser.add_argument('--es_patience', type=int, default=5,
                        help='Early Stopping patience.')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='The number of experiment seeds.')
    

    args = parser.parse_args()
    main(args)


                

