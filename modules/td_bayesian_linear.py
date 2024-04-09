from modules import NStepKLVCLBayesianLinear
import torch.nn.functional as F

class TDBayesianLinear(NStepKLVCLBayesianLinear):
    def __init__(self, input_size, output_size, n, lambd, lambda_logvar=-5.0):
        super(TDBayesianLinear, self).__init__(input_size, output_size, n, lambda_logvar)
        self.lambd = lambd

    def posterior_kl_div(self, curr_timestep):
        k = self.n - curr_timestep if curr_timestep < self.n else 1 # Decide the oldest posterior to consider
        discount = 1.0
        steps = 0.0
        kl_div = 0

        step_kl = self._kl_div(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights[self.n - 1].mean, self.prev_post_weights[self.n - 1].logvar) \
            + self._kl_div(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases[self.n - 1].mean, self.prev_post_biases[self.n - 1].logvar)
        steps += 1.0

        kl_div = kl_div + discount * step_kl
        
        for i in reversed(range(k, self.n - 1)):
            discount = self.lambd * discount
            step_kl = self._kl_div(self.posterior_weights.mean, self.posterior_weights.logvar, self.prev_post_weights[i].mean, self.prev_post_weights[i].logvar) \
                + self._kl_div(self.posterior_biases.mean, self.posterior_biases.logvar, self.prev_post_biases[i].mean, self.prev_post_biases[i].logvar)
            kl_div = kl_div + discount * step_kl
            steps += 1.0

        norm = (self.lambd - 1.0) / (self.lambd ** steps - 1.0)
        
        return kl_div * norm
        