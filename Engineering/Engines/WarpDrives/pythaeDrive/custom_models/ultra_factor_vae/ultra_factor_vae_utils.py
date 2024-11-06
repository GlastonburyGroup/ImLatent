import torch
import torch.nn as nn


class UltraFactorVAEDiscriminator(nn.Module):
    def __init__(self, latent_dim=16, hidden_units=1000) -> None:

        nn.Module.__init__(self)

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.PReLU(init=0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.PReLU(init=0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.PReLU(init=0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.PReLU(init=0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.PReLU(init=0.2),
            nn.Linear(hidden_units, 2),
        )

    def forward(self, z: torch.Tensor):
        return self.layers(z)

class UltraPredictor(nn.Module):
    def __init__(self, latent_dim=16, n_layers=2, n_predictions=2) -> None:

        nn.Module.__init__(self)
        
        assert n_layers >= 1, "n_layers must be at least 1"
        assert n_predictions >= 1, "n_predictions must be at least 1"
        assert latent_dim >= 1, "latent_dim must be at least 1"
        assert latent_dim % 2**(n_layers-1) == 0, "latent_dim must be divisible by 2 n_layers-1 times"

        if n_layers == 1:
            self.layers = nn.Linear(latent_dim, n_predictions)
        else:
            _layers = []
            for _ in range(n_layers-1):
                _layers.append(nn.Linear(latent_dim, latent_dim//2))
                latent_dim = latent_dim//2
                _layers.append(nn.PReLU(init=0.2))
            _layers.append(nn.Linear(latent_dim, n_predictions))
            self.layers = nn.Sequential(*_layers)

    def forward(self, z: torch.Tensor):
        return self.layers(z)
    
def pytcorr(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
    )

def anti_confounder_loss(z, confs):
    zeroT = torch.Tensor([0]).to(z.device)
    corr_loss = 0
    for cn in confs.split(1, dim=1):  
        for zj in z.split(1, dim=1):  
            corr_loss += torch.max(zeroT, pytcorr(zj.squeeze(), cn))