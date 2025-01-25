from torch import nn
import torch
import utils
from trainers import Trainer

class OnlineMLETrainer(Trainer):
    def __init__(self, model, args, device, weight_decay=0):
        super().__init__(model, args, device, weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, output, target):
        return self.loss_fn(output, target)

class MultiHeadOnlineMLETrainer(OnlineMLETrainer):
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