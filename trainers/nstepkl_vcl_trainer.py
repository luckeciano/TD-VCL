import torch
from trainers import Trainer
from data_structures import ReplayBuffer
import torch.nn.functional as F
import utils
import numpy as np

class NStepKLVCLTrainer(Trainer):
    def __init__(self, model, args, device, n, num_mem_tasks, task_mem_size, beta=5e-3, no_kl=False):
        super().__init__(model, args, device)
        self.no_kl = no_kl
        self.beta = beta
        self.replay_buffer = None # Lazy Construction
        self.n = n
        self.num_mem_tasks = num_mem_tasks
        self.task_mem_size = task_mem_size
        self.timestep = 0

    def compute_task_ll_weight(self, t):
        i = self.timestep - t
        i = torch.where(i < self.n - 1, i, self.n - 1) # in case of n < num_mem_tasks
        return (self.n - (i)) / self.n
    
    def compute_loss(self, output, target, t):
        b_size = output.shape[0]
        weight = self.compute_task_ll_weight(t)
        
        log_likelihood = - F.cross_entropy(output, target, reduction='none')
        weighted_ll = torch.mean(log_likelihood * weight)

        if self.no_kl:
            return -weighted_ll

        kl_div = self.model.kl_div(self.timestep)
        kl_div = kl_div / b_size # Since we are averaging log likelihood datapoints, we should divide kl_div by the batch_size

        loss = - (weighted_ll - self.beta*kl_div)

        return loss
    
    def set_timestep(self, t):
        self.timestep = t

    def add_to_buffer(self, x_train, y_train, t):
        if self.replay_buffer is None:
            self.replay_buffer = ReplayBuffer(self.num_mem_tasks, self.task_mem_size)
        self.replay_buffer.push(x_train, y_train, t)

    def train_eval_loop(self, task_generator, model, args, seed, break_search=False, break_search_min=0.5):
        x_test_sets = []
        y_test_sets = []
        test_accuracies = []
        test_accuracies_per_task = {i: [] for i in range(task_generator.max_iter)}

        for task_id in range(task_generator.max_iter):
            model.new_task(task_id, args.single_head)
            self.set_timestep(task_id)
            
            x_train, y_train, x_test, y_test = task_generator.next_task()
            t_train = self.timestep * np.ones(shape=(len(y_train), 1))
            x_test_sets.append(x_test)
            y_test_sets.append(y_test)
            test_dataloaders = utils.get_dataloaders(x_test_sets, y_test_sets, args.batch_size, shuffle=False, drop_last=False)
            
            self.train(args.epochs_per_task, args, seed, x_train, y_train, t_train)

            print(f"Test Accuracy after task {task_id}:")
            acc, acc_tasks = self.evaluate(test_dataloaders, single_head=args.single_head)
            if break_search and min(acc_tasks) < break_search_min:
                return None, None
            test_accuracies.append(acc)
            for idx, task_acc in enumerate(acc_tasks):
                test_accuracies_per_task[idx].append(task_acc)
            
            self.add_to_buffer(x_train, y_train, t_train)
        
        return test_accuracies, test_accuracies_per_task
    
    def train(self, epochs, args, seed, x_train, y_train, t_train):
        current_task_train_size = int(len(x_train)  * (1.0 - args.valid_ratio))
        final_x, final_y, final_t, valid_loader = utils.get_nstepkl_data(self.replay_buffer, self.n, x_train, y_train, t_train, args, seed)
        if self.es:
            self.early_stop.reset()

        for epoch in range(epochs):
            self.model.train()
            train_loader = utils.get_nstepkl_trainloader(final_x, final_y, final_t, args.batch_size, current_task_train_size, shuffle=True, drop_last=True)
            for inputs, targets, t in train_loader:
                inputs, targets, t = inputs.to(self.device), targets.to(self.device), t.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets, t)
                acc, _, _ = self.compute_metrics(outputs, targets)
                # print(f'Epoch: {epoch} Train loss: {loss.item()} Batch Accuracy: {acc} ')
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()

            train_loss = 0
            self.model.eval()
            train_corrects = 0
            train_total = 0
            with torch.no_grad():
                for inputs, targets, t in train_loader:
                    inputs, targets, t = inputs.to(self.device), targets.to(self.device), t.to(self.device)
                    outputs = self.model(inputs, sample=False)
                    train_loss += self.compute_loss(outputs, targets, t)
                    _, corr, tot = self.compute_metrics(outputs, targets)
                    train_corrects += corr
                    train_total += tot
            
            train_acc = train_corrects / train_total

            valid_loss = 0
            self.model.eval()
            valid_corrects = 0
            valid_total = 0
            with torch.no_grad():
                for inputs, targets, t in valid_loader:
                    inputs, targets, t = inputs.to(self.device), targets.to(self.device), t.to(self.device)
                    outputs = self.model(inputs, sample=False)
                    valid_loss += self.compute_loss(outputs, targets, t)
                    _, corr, tot = self.compute_metrics(outputs, targets)
                    valid_corrects += corr
                    valid_total += tot
            
            valid_acc = valid_corrects / valid_total

            if self.es:
                self.early_stop(valid_loss)
                if self.early_stop.early_stop:
                    print(f"FinalTraining Accuracy: {train_acc}")
                    print(f"Final Validation Accuracy: {valid_acc}")
                    self.early_stop.reset()
                    break

    def evaluate(self, eval_loaders, single_head):
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        acc_tasks = []

        for idx, eval_loader in enumerate(eval_loaders):
            self.model.set_task(0 if single_head else idx)
            task_correct = 0
            task_total = 0
            with torch.no_grad():
                for inputs, targets in eval_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs, sample=False)
                    _, correct_batch, total_batch = self.compute_metrics(outputs, targets)
                    correct_predictions += correct_batch
                    total_predictions += total_batch
                    task_correct += correct_batch
                    task_total += total_batch
                
                acc_task = task_correct / task_total
                acc_tasks.append(acc_task)
        
        acc = correct_predictions / total_predictions
        print(f'Accuracy: {acc}')
        print(f'Accuracy for each Task: {acc_tasks}')
        return acc, acc_tasks
    

class MultiHeadNStepKLVCLTrainer(NStepKLVCLTrainer):
    def _select_outputs(self, output, target):
        target_values = target[:, :-1]
        task_ids = target[:, -1].long()

        selected_outputs = output[torch.arange(output.shape[0]), task_ids, :]
        return target_values, selected_outputs

    def compute_loss(self, output, target, t):
        target_values, selected_outputs = self._select_outputs(output, target)
        loss = super().compute_loss(selected_outputs, target_values, t)
        return loss
    
    def compute_metrics(self, output, target):
        target_values, selected_outputs = self._select_outputs(output, target)
        return utils.compute_accuracy(selected_outputs, target_values)