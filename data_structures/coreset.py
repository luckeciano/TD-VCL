import random
import numpy as np

def get_random_coreset(x_train, y_train, coresets, K):
    indices = list(range(x_train.shape[0]))
    indices = random.sample(indices, K)

    task_coreset_x = x_train[indices]
    task_coreset_y = y_train[indices]

    updated_x_train = np.delete(x_train, indices, axis=0)
    updated_y_train = np.delete(y_train, indices, axis=0)

    old_coreset_x, old_coreset_y = coresets[0], coresets[1]
    
    for x_task, y_task in zip(old_coreset_x, old_coreset_y):
        updated_x_train = np.vstack((updated_x_train, x_task))
        updated_y_train = np.vstack((updated_y_train, y_task))

    old_coreset_x.append(task_coreset_x)
    old_coreset_y.append(task_coreset_y)

    return updated_x_train, updated_y_train, (old_coreset_x, old_coreset_y)


def select_memory_set(x_train_sets, y_train_sets, num_tasks_mem, task_mem_size, ft_size, num_classes):
    tasks_sampled = 0
    mem_x_train = np.empty((0, *ft_size))
    mem_y_train = np.array([]).reshape(0, num_classes)
    for i in range(len(x_train_sets) - 1, -1, -1):
        mem_task_x = x_train_sets[i]
        mem_task_y = y_train_sets[i]
        indices = list(range(mem_task_y.shape[0]))
        indices = random.sample(indices, task_mem_size)

        mem_x_train = np.vstack((mem_x_train, mem_task_x[indices]))
        mem_y_train = np.vstack((mem_y_train, mem_task_y[indices]))

        tasks_sampled += 1
        if tasks_sampled >= num_tasks_mem:
            break
    
    return mem_x_train, mem_y_train
