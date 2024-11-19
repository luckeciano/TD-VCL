import os
import numpy as np
import torch
from sklearn.utils import shuffle
import gzip
import pickle

class SplitCIFAR100():
    def __init__(self, task_id=False):
        self.num_classes = 10
        self.num_tasks = 10
        self.size = [3, 32, 32]

        f = gzip.open('data/split_cifar100.pkl.gz', 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        self.train, self.test = u.load()
        f.close()

        self.task_id = task_id
        self.max_iter = len(self.train)
        self.cur_iter = 0
    

    def get_dims(self):
        # Get data input and output dimensions
        return self.size, self.num_classes

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            next_x_train = np.squeeze(np.stack(self.train[self.cur_iter]['x'], axis=0))
            next_y_train = np.eye(self.num_classes)[np.stack(self.train[self.cur_iter]['y'], axis=0)]

            next_x_test = np.squeeze(np.stack(self.test[self.cur_iter]['x'], axis=0))
            next_y_test = np.eye(self.num_classes)[np.stack(self.test[self.cur_iter]['y'], axis=0)]

            if self.task_id:
                task_ids_train = np.full((next_y_train.shape[0], 1), self.cur_iter)
                next_y_train = np.hstack((next_y_train, task_ids_train))

                task_ids_test = np.full((next_y_test.shape[0], 1), self.cur_iter)
                next_y_test = np.hstack((next_y_test, task_ids_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test
