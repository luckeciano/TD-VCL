import torch
import numpy as np
import pandas as pd
import random
import sklearn.model_selection as sk
from torch.utils.data import TensorDataset, DataLoader

def train_test_split(x, y, t, ratio, seed):
    combined_arrays = list(zip(x, y, t))
    random.shuffle(combined_arrays)
    split_index = int(len(x) * ratio)

    x, y, t = zip(*combined_arrays)
    x_train, x_valid = x[split_index:], x[:split_index]
    y_train, y_valid = y[split_index:], y[:split_index]
    t_train, t_valid = t[split_index:], t[:split_index]

    return np.array(x_train), np.array(x_valid), np.array(y_train), np.array(y_valid), np.array(t_train), np.array(t_valid)

def generate_data_splits(task_gen, x_test_sets, y_test_sets, seed, valid_ratio, coresets=None, coreset_method=None, K=0):
    x_train, y_train, x_test, y_test = task_gen.next_task()

    new_coresets = ()
    if coresets:
        x_train, y_train, new_coresets = coreset_method(x_train, y_train, coresets, K)

    x_test_sets.append(x_test)
    y_test_sets.append(y_test)

    x_train, x_valid, y_train, y_valid = sk.train_test_split(x_train, y_train, test_size=valid_ratio, random_state=seed)
    return x_train, y_train, x_valid, y_valid, new_coresets


def generate_dataloaders(x_train, y_train, x_valid, y_valid, x_test_sets, y_test_sets, batch_size, seed):
    train_dataloader = get_dataloader(x_train, y_train, batch_size, shuffle=True)
    valid_dataloader = get_dataloader(x_valid, y_valid, batch_size, shuffle=True)

    test_dataloaders = get_dataloaders(x_test_sets, y_test_sets, batch_size, shuffle=False, drop_last=False)

    return train_dataloader, valid_dataloader, test_dataloaders

def get_dataloader(x, y, batch_size, shuffle=False, drop_last=True):
    tensor_x = torch.Tensor(x) 
    tensor_y = torch.Tensor(y)

    # Create a TensorDataset from tensors
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_nstepkl_trainloader(x, y, t, batch_size, epoch_size, shuffle=False, drop_last=True):
    indices = list(range(x.shape[0]))
    indices = random.sample(indices, epoch_size)

    return get_nstepkl_dataloader(x[indices], y[indices], t[indices], batch_size, shuffle, drop_last)

def get_nstepkl_data(replay_buffer, n, task_x, task_y, task_t, args, seed, shuffle=False, drop_last=True):
    if replay_buffer:
        x, y, t = replay_buffer.retrieve(n, len(task_x))
        
        x = np.vstack((x, task_x))
        y = np.vstack((y, task_y))
        t = np.vstack((t, task_t))
    else:
        x, y, t = task_x, task_y, task_t

    x_train, x_valid, y_train, y_valid, t_train, t_valid = train_test_split(x, y, t, args.valid_ratio, seed)

    valid_dataloader = get_nstepkl_dataloader(x_valid, y_valid, t_valid, args.batch_size, shuffle, drop_last)
    return x_train, y_train, t_train, valid_dataloader


def get_nstepkl_dataloader(x, y, t, batch_size, shuffle=False, drop_last=True):
    tensor_x = torch.Tensor(x) 
    tensor_y = torch.Tensor(y)
    tensor_t = torch.Tensor(t)

    # Create a TensorDataset from tensors
    dataset = TensorDataset(tensor_x, tensor_y, tensor_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_dataloaders(x_list, y_list, batch_size, shuffle, drop_last):
    dataloaders = []
    for x, y in zip(x_list, y_list):
        dataloaders.append(get_dataloader(x, y, batch_size, shuffle, drop_last))
    
    return dataloaders

def compute_accuracy(logits, targets):
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    targets_argmax  = torch.argmax(targets, dim=1)
    correct_predictions = (predictions == targets_argmax).sum().item()
    total_predictions = targets.size(0)
    acc = correct_predictions / total_predictions
    return acc, correct_predictions, total_predictions