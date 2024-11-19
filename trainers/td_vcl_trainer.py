from trainers import NStepKLVCLTrainer
import torch
import utils

class TemporalDifferenceVCLTrainer(NStepKLVCLTrainer):
    def __init__(self, model, args, device, n, lambd, num_mem_tasks, task_mem_size, beta=5e-3, no_kl=False):
        super().__init__(model, args, device, n, num_mem_tasks, task_mem_size, beta, no_kl)
        self.lambd = lambd

    def compute_task_ll_weight(self, t):
        i = self.timestep - t
        i = torch.where(i < self.n - 1, i, self.n - 1) # in case of n < num_mem_tasks
        return ((self.lambd**i) * (self.lambd**(self.n - i) - 1.0)) / (self.lambd**(self.n) - 1.0)
    

class MultiHeadTDVCLTrainer(TemporalDifferenceVCLTrainer):
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