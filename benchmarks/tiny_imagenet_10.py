import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class TinyImagenet(Dataset):
    """Defines the Tiny Imagenet dataset."""

    def __init__(self, task_id = False) -> None:
        self.num_classes = 20
        self.num_tasks = 10
        self.size = [3, 64, 64]

        self.x_train = []
        self.x_test = []
        for num in range(20):
            self.x_train.append(np.load('data/processed/x_train_%02d.npy' %
                      (num + 1)))
            self.x_test.append(np.load('data/processed/x_val_%02d.npy' %
                      (num + 1)))
        self.x_train = np.concatenate(np.array(self.x_train))
        self.x_test = np.concatenate(np.array(self.x_test))

        self.y_train = []
        self.y_test = []
        for num in range(20):
            self.y_train.append(np.load('data/processed/y_train_%02d.npy' %
                      (num + 1)))
            self.y_test.append(np.load('data/processed/y_val_%02d.npy' %
                      (num + 1)))
        self.y_train = np.concatenate(np.array(self.y_train))
        self.y_test = np.concatenate(np.array(self.y_test))

        self.tasks = self.split_tasks(self.num_tasks, seed=42)
        self.task_id = task_id
        self.max_iter = len(self.tasks)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.size, self.num_classes

    def split_tasks(self, num_tasks=10, seed=None):
        """
        Splits the dataset into tasks based on class labels.

        Args:
            num_tasks (int): Number of tasks to split the dataset into.
            seed (int): Seed for randomization to ensure reproducibility.
        Returns:
            List of tuples: Each tuple contains train and test data for a task with relabeled targets.
        """
        unique_classes = np.unique(self.y_train)  # Get unique class labels
        assert len(unique_classes) % num_tasks == 0, "Classes cannot be evenly split into tasks."
        classes_per_task = len(unique_classes) // num_tasks
        
        # Set the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Shuffle the classes to ensure random task assignment
        np.random.shuffle(unique_classes)
        
        tasks = []
        for i in range(num_tasks):
            task_classes = unique_classes[i * classes_per_task:(i + 1) * classes_per_task]
            task_indices_train = np.isin(self.y_train, task_classes)
            task_indices_test = np.isin(self.y_test, task_classes)

            x_train_task = self.x_train[task_indices_train]
            y_train_task = self.y_train[task_indices_train]
            x_test_task = self.x_test[task_indices_test]
            y_test_task = self.y_test[task_indices_test]

            # Relabel the targets for the task to range from 0 to 19
            relabel_map = {original: new for new, original in enumerate(task_classes)}
            y_train_task = np.vectorize(relabel_map.get)(y_train_task)
            y_test_task = np.vectorize(relabel_map.get)(y_test_task)

            # Shuffle the dataset for each task
            train_indices = np.random.permutation(len(x_train_task))
            x_train_task = x_train_task[train_indices]
            y_train_task = y_train_task[train_indices]

            test_indices = np.random.permutation(len(x_test_task))
            x_test_task = x_test_task[test_indices]
            y_test_task = y_test_task[test_indices]

            # Change data format to (3, 64, 64) - channels first
            x_train_task = np.transpose(x_train_task, (0, 3, 1, 2))  # Transpose to (3, 64, 64)
            x_test_task = np.transpose(x_test_task, (0, 3, 1, 2))

            tasks.append((x_train_task, y_train_task, x_test_task, y_test_task))
        return tasks
    
    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            next_x_train, next_y_train, next_x_test, next_y_test = self.tasks[self.cur_iter]
            next_y_train = np.eye(self.num_classes)[next_y_train]

            next_y_test = np.eye(self.num_classes)[next_y_test]

            if self.task_id:
                task_ids_train = np.full((next_y_train.shape[0], 1), self.cur_iter)
                next_y_train = np.hstack((next_y_train, task_ids_train))

                task_ids_test = np.full((next_y_test.shape[0], 1), self.cur_iter)
                next_y_test = np.hstack((next_y_test, task_ids_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


    