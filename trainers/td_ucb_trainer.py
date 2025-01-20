from trainers import TemporalDifferenceVCLTrainer, UCBOptimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import utils

class TemporalDifferenceUCBTrainer(TemporalDifferenceVCLTrainer):
    def __init__(self, model, args, device, n, lambd, num_mem_tasks, task_mem_size, alpha, beta=5e-3, no_kl=False, weight_decay=0, upsample=True):
        super().__init__(model, args, device, n, lambd, num_mem_tasks, task_mem_size, beta, no_kl, upsample)
        self.lambd = lambd
        self.optimizer = UCBOptimizer(self.model.named_parameters(), lr=args.lr, alpha=alpha, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs_per_task)
        self.optimizer.scheduler = self.scheduler
    

class MultiHeadTDUCBTrainer(TemporalDifferenceUCBTrainer):
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