from trainers import Trainer
import torch.nn.functional as F

class VCLTrainer(Trainer):
    def __init__(self, model, args, device, beta=5e-3, no_kl=False):
        super().__init__(model, args, device)
        self.no_kl = no_kl
        self.beta = beta

    def compute_loss(self, output, target):
        b_size = output.shape[0]
        log_likelihood = - F.cross_entropy(output, target)

        if self.no_kl:
            return -log_likelihood

        kl_div = self.model.kl_div()
        kl_div = kl_div / b_size # Since we are averaging log likelihood datapoints, we should divide kl_div by the batch_size

        loss = - (log_likelihood - self.beta*kl_div)

        return loss