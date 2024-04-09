from trainers import VCLTrainer
from sklearn.model_selection import train_test_split
from collections import deque
import numpy as np
import utils
import copy

class VCLCoreSetTrainer(VCLTrainer):
    def __init__(self, model, args, device, beta=5e-3, no_kl=False, coreset_method=None, K=0, max_tasks=10):
        super().__init__(model, args, device)
        self.no_kl = no_kl
        self.beta = beta
        self.coreset_method = coreset_method
        self.K = K
        self.max_tasks = max_tasks

    def train_eval_loop(self, task_generator, model, args, seed):
        x_test_sets = []
        y_test_sets = []
        test_accuracies = []
        test_accuracies_per_task = {i: [] for i in range(task_generator.max_iter)}
        ft_size, num_classes = task_generator.get_dims()
        coresets = (deque(maxlen=self.max_tasks), deque(maxlen=self.max_tasks)) # x and y coresets

        for task_id in range(task_generator.max_iter):
            model.new_task(task_id, args.single_head)
            x_train, y_train, x_valid, y_valid, coresets = utils.generate_data_splits(task_generator, x_test_sets, y_test_sets, seed, args.valid_ratio, coresets, self.coreset_method, self.K)
            train_dataloader, valid_dataloader, test_dataloaders = utils.generate_dataloaders(x_train, y_train, x_valid, y_valid, x_test_sets, y_test_sets, args.batch_size, seed)
            
            self.train(args.epochs_per_task, train_dataloader, valid_dataloader)

            # Adaptation
            pred_model = copy.deepcopy(self)

            coreset_x_train, coreset_y_train = np.empty((0, ft_size)), np.empty((0, num_classes))
            for x_task, y_task in zip(coresets[0], coresets[1]):
                coreset_x_train = np.vstack((coreset_x_train, x_task))
                coreset_y_train = np.vstack((coreset_y_train, y_task))
            
            x_train, x_valid, y_train, y_valid = train_test_split(coreset_x_train, coreset_y_train, test_size=args.valid_ratio, random_state=seed)
            train_dataloader = utils.get_dataloader(x_train, y_train, args.batch_size, shuffle=True, drop_last=False)
            valid_dataloader = utils.get_dataloader(x_valid, y_valid, args.batch_size, shuffle=True, drop_last=False)
            pred_model.train(args.epochs_per_task, train_dataloader, valid_dataloader)

            print(f"Test Accuracy after task {task_id}:")
            acc, acc_tasks = self.evaluate(test_dataloaders, single_head=args.single_head)
            test_accuracies.append(acc)
            for idx, task_acc in enumerate(acc_tasks):
                test_accuracies_per_task[idx].append(task_acc)
            
            del pred_model
        
        return test_accuracies, test_accuracies_per_task