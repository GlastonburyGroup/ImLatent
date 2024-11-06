import torch

class ChannelVarWeightedLoss(torch.nn.Module):
    def __init__(self, loss_fn="mse"):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, input, target):
        weights = 1+torch.var(target, dim=1, keepdim=True)
        if self.loss_fn == "mse":
            loss = weights * (target - input) ** 2
        elif self.loss_fn == "mae" or self.loss_fn == "l1":
            loss = weights * torch.abs(target - input)
        return loss.reshape(target.shape[0], -1).sum(dim=-1)