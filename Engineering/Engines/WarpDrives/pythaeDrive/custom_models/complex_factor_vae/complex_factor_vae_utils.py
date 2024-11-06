import torch
import torch.nn as nn
import torchcomplex.nn as cnn

class CVFactorVAEDiscriminator(nn.Module):
    def __init__(self, latent_dim=16, hidden_units=1000) -> None:

        nn.Module.__init__(self)

        self.layers = nn.Sequential(
            cnn.Linear(latent_dim, hidden_units),
            cnn.modReLU(),
            cnn.Linear(hidden_units, hidden_units),
            cnn.modReLU(),
            cnn.Linear(hidden_units, hidden_units),
            cnn.modReLU(),
            cnn.Linear(hidden_units, hidden_units),
            cnn.modReLU(),
            cnn.Linear(hidden_units, hidden_units),
            cnn.modReLU(),
            cnn.Linear(hidden_units, 2),
            cnn.modSigmoid()
        )

    def forward(self, z: torch.Tensor):
        return self.layers(z)