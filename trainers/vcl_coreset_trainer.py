from trainers import VCLTrainer
from sklearn.model_selection import train_test_split
from collections import deque
import numpy as np
import torch
from training_utils.early_stopping import EarlyStopping
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

            coreset_x_train, coreset_y_train = np.empty((0, *ft_size)), np.empty((0, num_classes))
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
    
class MultiHeadVCLCoreSetTrainer(VCLCoreSetTrainer):
    def __init__(self, model, args, device, beta=5e-3, no_kl=False, coreset_method=None, K=0, max_tasks=10):
        super(MultiHeadVCLCoreSetTrainer, self).__init__(model, args, device, beta, no_kl, coreset_method, K, max_tasks)

    def _select_outputs(self, output, target):
        target_values = target[:, :-1]
        task_ids = target[:, -1].long()

        selected_outputs = output[torch.arange(output.shape[0]), task_ids, :]
        return target_values, selected_outputs

    def compute_loss_coreset(self, output, target):
        target_values, selected_outputs = self._select_outputs(output, target)
        loss = torch.nn.CrossEntropyLoss()(selected_outputs, target_values)
        return loss
    
    def compute_loss_vcl(self, output, target):
        target_values, selected_outputs = self._select_outputs(output, target)
        return super().compute_loss(selected_outputs, target_values)
    
    def compute_metrics(self, output, target):
        target_values, selected_outputs = self._select_outputs(output, target)
        return utils.compute_accuracy(selected_outputs, target_values)
        
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
            
            self.train(args.epochs_per_task, train_dataloader, valid_dataloader, loss_fn=self.compute_loss_vcl)

            # Adaptation
            pred_model = copy.deepcopy(self)
            if pred_model.es:
                pred_model.early_stop = EarlyStopping(patience=3)

            coreset_x_train, coreset_y_train = np.empty((0, *ft_size)), np.empty((0, num_classes + 1))
            for x_task, y_task in zip(coresets[0], coresets[1]):
                coreset_x_train = np.vstack((coreset_x_train, x_task))
                coreset_y_train = np.vstack((coreset_y_train, y_task))

            if len(coreset_x_train) != 0:  
                x_train, x_valid, y_train, y_valid = train_test_split(coreset_x_train, coreset_y_train, test_size=args.valid_ratio, random_state=seed)
                train_dataloader = utils.get_dataloader(x_train, y_train, args.batch_size, shuffle=True, drop_last=False)
                valid_dataloader = utils.get_dataloader(x_valid, y_valid, args.batch_size, shuffle=True, drop_last=False)
                pred_model.train(args.epochs_per_task, train_dataloader, valid_dataloader, loss_fn=self.compute_loss_coreset)

            print(f"Test Accuracy after task {task_id}:")
            acc, acc_tasks = pred_model.evaluate(test_dataloaders, single_head=args.single_head)
            test_accuracies.append(acc)
            for idx, task_acc in enumerate(acc_tasks):
                test_accuracies_per_task[idx].append(task_acc)
            
            del pred_model
        
        return test_accuracies, test_accuracies_per_task