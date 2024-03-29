import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def get_dataloader(x, y, batch_size, shuffle=False, drop_last=True):
    tensor_x = torch.Tensor(x) 
    tensor_y = torch.Tensor(y)

    # Create a TensorDataset from tensors
    dataset = TensorDataset(tensor_x, tensor_y)
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