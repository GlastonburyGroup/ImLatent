import torch

class MaskedReconLoss(torch.nn.Module):
    def __init__(self, loss_fn="mse", mode=0):
        super().__init__()
        self.loss_fn = loss_fn
        self.mode = mode #mode: 0 = only masked loss, 1: masked loss + unmasked loss

    def forward(self, input, target, mask):
        masked_target = target * mask
        masked_input = input * mask
        if self.loss_fn == "mse":
            masked_loss = (masked_target - masked_input) ** 2
        elif self.loss_fn in ["mae", "l1"]:
            masked_loss = torch.abs(masked_target - masked_input)
        if str(self.mode) == "0":
            return masked_loss.reshape(target.shape[0], -1).sum(dim=-1)
        elif str(self.mode) == "1":
            if self.loss_fn == "mse":
                unmasked_loss = (target - input) ** 2
            elif self.loss_fn in ["mae", "l1"]:
                unmasked_loss = torch.abs(target - input)
            return masked_loss.reshape(target.shape[0], -1).sum(dim=-1) + unmasked_loss.reshape(target.shape[0], -1).sum(dim=-1)
