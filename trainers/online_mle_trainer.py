from torch import nn
from trainers import Trainer

class OnlineMLETrainer(Trainer):
    def __init__(self, model, args, device, weight_decay=0):
        super().__init__(model, args, device, weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, output, target):
        return self.loss_fn(output, target)