import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from trainers import VCLTrainer
import math


class UCBOptimizer(torch.optim.Optimizer):
    
    def __init__(self, named_params, lr, alpha, *args, **kwargs):
        """
        Custom optimizer that temporarily adjusts the learning rate based on the weights.
        
        Args:
            params: Model parameters.
            lr: Global learning rate.
        """
         # Initialize empty lists for mean and logvar parameter groups
        self.mean_params_groups = []
        self.logvar_params_groups = []

        # Group parameters based on "mean" or "logvar"
        for name, param in named_params:
            if "mean" in name:
                self.mean_params_groups.append({"params": [param], "lr": lr})  # Each mean param gets its own group
            elif "logvar" in name:
                self.logvar_params_groups.append({"params": [param], "lr": lr})  # Each logvar param gets its own group

        defaults = dict(lr=lr, betas=(0.9, 0.999), eps=1e-8,
                        weight_decay= 0, amsgrad=False,
                        maximize=False, foreach=None, capturable=False,
                        differentiable=False, fused=False)
        
        # Update logvar params with standard Adam
        super().__init__(self.logvar_params_groups + self.mean_params_groups, defaults)
        self.global_lr = lr
        self.alpha = alpha

    def step(self, closure=None):
        """
        Perform a single optimization step with adjusted learning rates.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        if closure is not None:
            closure()

        # Update the global learning rate if a scheduler is present
        if self.scheduler is not None:
            self.global_lr = self.scheduler.get_last_lr()[0]

        # Temporarily adjust learning rates
        for mean_group, logvar_group in zip(self.mean_params_groups, self.logvar_params_groups):
            mean_param = mean_group['params'][0]
            logvar_param = logvar_group['params'][0]
            sigma = torch.exp(0.5 * logvar_param)
            adjusted_lr = self.global_lr * sigma * self.alpha
            self._grad_update(mean_param, adjusted_lr)
            self._grad_update(logvar_param, self.global_lr)

    
    def _grad_update(self, mean_param, adjusted_lr):
        # Adam state management
        if mean_param.grad is not None:
            state = self.state[mean_param]
            if len(state) == 0:
                # Lazy initialization of Adam state
                state['step'] = 0.0
                state['exp_avg'] = torch.zeros_like(mean_param.data)
                state['exp_avg_sq'] = torch.zeros_like(mean_param.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.defaults['betas']

            # Update step count
            state['step'] += 1
            step = state['step']

            # Update biased first and second moment estimates
            exp_avg.mul_(beta1).add_(mean_param.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(mean_param.grad, mean_param.grad, value=1 - beta2)

            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            adjusted_lr = adjusted_lr / bias_correction1
            sqrt_bc2 = math.sqrt(bias_correction2)
            sqrt_exp_avg_sq = torch.sqrt(exp_avg_sq)

            sqrt_div = sqrt_exp_avg_sq / sqrt_bc2

            corrected_lr = adjusted_lr / (sqrt_div + self.defaults['eps'])

            # Apply parameter update
            mean_param.data -= corrected_lr * exp_avg


class UCBTrainer(VCLTrainer):
    def __init__(self, model, args, device, beta=5e-3, alpha=1.0, no_kl=False, weight_decay=0):
        super().__init__(model, args, device, beta, no_kl)
        self.no_kl = no_kl
        self.beta = beta
        self.optimizer = UCBOptimizer(self.model.named_parameters(), lr=args.lr, alpha=alpha, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs_per_task)
        self.optimizer.scheduler = self.scheduler


