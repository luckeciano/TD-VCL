from torch import nn
from trainers import OnlineMLETrainer
from sklearn.model_selection import train_test_split
import numpy as np
import utils
from data_structures import select_memory_set

class BatchMLETrainer(OnlineMLETrainer):
    def __init__(self, model, args, device, num_tasks_mem, task_mem_size, weight_decay=0):
        super().__init__(model, args, device, weight_decay)
        self.num_tasks_mem = num_tasks_mem
        self.task_mem_size = task_mem_size

    def train_eval_loop(self, task_generator, model, args, seed):
        x_test_sets = []
        y_test_sets = []
        x_train_sets = []
        y_train_sets = []
        test_accuracies = []
        test_accuracies_per_task = {i: [] for i in range(task_generator.max_iter)}

        for task_id in range(task_generator.max_iter):
            model.new_task(task_id, args.single_head)
            ft_size, num_classes = task_generator.get_dims()
            mem_x_train, mem_y_train = select_memory_set(x_train_sets, y_train_sets, self.num_tasks_mem, self.task_mem_size, ft_size, num_classes)
            task_x_train, task_y_train, x_test, y_test = task_generator.next_task()

            x_train = np.vstack((task_x_train, mem_x_train))
            y_train = np.vstack((task_y_train, mem_y_train))

            x_test_sets.append(x_test)
            y_test_sets.append(y_test)
            x_train_sets.append(task_x_train)
            y_train_sets.append(task_y_train)
            
            x_train, x_valid, y_train, y_valid = train_test_split(np.array(x_train), np.array(y_train), test_size=args.valid_ratio, random_state=seed)

            train_dataloader, valid_dataloader, test_dataloaders = utils.generate_dataloaders(x_train, y_train, x_valid, y_valid, x_test_sets, y_test_sets, args.batch_size, seed)
            
            self.train(args.epochs_per_task, train_dataloader, valid_dataloader)

            print(f"Test Accuracy after task {task_id}:")
            acc, acc_tasks = self.evaluate(test_dataloaders, single_head=args.single_head)
            test_accuracies.append(acc)
            for idx, task_acc in enumerate(acc_tasks):
                test_accuracies_per_task[idx].append(task_acc)
        
        return test_accuracies, test_accuracies_per_task
        