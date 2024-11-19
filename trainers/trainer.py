import torch
from torch import nn
import utils
from torch.optim.lr_scheduler import CosineAnnealingLR
from training_utils import EarlyStopping

class Trainer:
    def __init__(self, model, args, device, weight_decay=0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs_per_task)
        self.es = args.enable_early_stopping

        if self.es:
            self.early_stop = EarlyStopping(patience=args.es_patience)

    def compute_loss(self, output, target):
        loss = nn.CrossEntropyLoss()(output, target)
        return loss
    
    def compute_metrics(self, output, target):
        return utils.compute_accuracy(output, target)
    
    def train_eval_loop(self, task_generator, model, args, seed):
        x_test_sets = []
        y_test_sets = []
        test_accuracies = []
        test_accuracies_per_task = {i: [] for i in range(task_generator.max_iter)}

        for task_id in range(task_generator.max_iter):
            model.new_task(task_id, args.single_head)
            x_train, y_train, x_valid, y_valid, _ = utils.generate_data_splits(task_generator, x_test_sets, y_test_sets, seed, args.valid_ratio)
            train_dataloader, valid_dataloader, test_dataloaders = utils.generate_dataloaders(x_train, y_train, x_valid, y_valid, x_test_sets, y_test_sets, args.batch_size, seed)
            
            self.train(args.epochs_per_task, train_dataloader, valid_dataloader)

            print(f"Test Accuracy after task {task_id}:")
            acc, acc_tasks = self.evaluate(test_dataloaders, single_head=args.single_head)
            test_accuracies.append(acc)
            for idx, task_acc in enumerate(acc_tasks):
                test_accuracies_per_task[idx].append(task_acc)
        
        return test_accuracies, test_accuracies_per_task

    def train(self, epochs, train_loader, valid_loader):
        if self.es:
            self.early_stop.reset()

        for epoch in range(epochs):
            self.model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)
                acc, _, _  = self.compute_metrics(outputs, targets)
                # print(f'Epoch: {epoch} Train loss: {loss.item()} Batch Accuracy: {acc} ')
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()

            train_loss = 0
            self.model.eval()
            train_corrects = 0
            train_total = 0
            with torch.no_grad():
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs, sample=False)
                    train_loss += self.compute_loss(outputs, targets)
                    _, corr, tot = self.compute_metrics(outputs, targets)
                    train_corrects += corr
                    train_total += tot
            
            train_acc = train_corrects / train_total

            valid_loss = 0
            self.model.eval()
            valid_corrects = 0
            valid_total = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs, sample=False)
                    valid_loss += self.compute_loss(outputs, targets)
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
        
