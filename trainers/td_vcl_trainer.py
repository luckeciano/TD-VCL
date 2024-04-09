from trainers import NStepKLVCLTrainer
import torch

class TemporalDifferenceVCLTrainer(NStepKLVCLTrainer):
    def __init__(self, model, args, device, n, lambd, num_mem_tasks, task_mem_size, beta=5e-3, no_kl=False):
        super().__init__(model, args, device, n, num_mem_tasks, task_mem_size, beta, no_kl)
        self.lambd = lambd

    def compute_task_ll_weight(self, t):
        i = self.timestep - t
        i = torch.where(i < self.n - 1, i, self.n - 1) # in case of n < num_mem_tasks
        return ((self.lambd**i) * (self.lambd**(self.n - i) - 1.0)) / (self.lambd**(self.n) - 1.0)