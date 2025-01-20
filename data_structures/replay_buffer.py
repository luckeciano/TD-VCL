from collections import deque
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, num_tasks, task_size, upsample=True):
        self.num_tasks = num_tasks
        self.task_size = task_size
        self.buffer_x = deque(maxlen=num_tasks)
        self.buffer_y = deque(maxlen=num_tasks)
        self.buffer_t = deque(maxlen=num_tasks)
        self.upsample = upsample

    def push(self, x_train, y_train, t):
        self.buffer_x.append(x_train)
        self.buffer_y.append(y_train)
        self.buffer_t.append(t)

    def retrieve(self, n, upsample_weight):
        x_train = []
        y_train = []
        t_train = []

        tasks_sampled = 0
        for x_task, y_task, t in zip(reversed(self.buffer_x), reversed(self.buffer_y), reversed(self.buffer_t)):
            indices = list(range(x_task.shape[0]))
            indices = random.sample(indices, self.task_size)
            task_x = x_task[indices]
            task_y = y_task[indices]
            task_t = t[indices]

            if self.upsample:
                task_x = np.repeat(task_x, int(upsample_weight / len(task_x)), axis=0)
                task_y = np.repeat(task_y, int(upsample_weight / len(task_y)), axis=0)
                task_t = np.repeat(task_t, int(upsample_weight / len(task_t)), axis=0)
            
            x_train.append(task_x)
            y_train.append(task_y)
            t_train.append(task_t)

            tasks_sampled += 1
            if tasks_sampled >= n:
                break

        return np.vstack(x_train), np.vstack(y_train), np.vstack(t_train)

    def __len__(self):
        return len(self.buffer_x)

