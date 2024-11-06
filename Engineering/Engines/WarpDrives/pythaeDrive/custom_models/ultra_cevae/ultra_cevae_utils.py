import random
import numpy as np

import torch
import torch.nn as nn

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

class CESquareNoiseGenerator:
    def __init__(self, dim, square_size, n_squares, square_pos=None, rnd_type="uniform"):
        self.square_size = square_size
        self.n_squares = n_squares
        self.square_pos = square_pos
        self.dim = dim
        self.rnd_type = rnd_type

        if self.dim == 2:
            self.generator = self.get_square_mask_2D
        elif self.dim == 3:
            self.generator = self.get_square_mask_3D

    def __call__(self, data_shape, dtype, device, noise_val):
        ret_data = torch.zeros(data_shape, dtype=dtype, device=device)
        for sample_idx in range(data_shape[0]):
            rnd_square_size = self._get_range_val(self.square_size)
            rnd_n_squares = self._get_range_val(self.n_squares)
            for _ in range(rnd_n_squares):
                ret_data[sample_idx] += self.generator(
                    dtype=dtype,
                    device=device,
                    data_shape=data_shape[1:],
                    square_size=rnd_square_size,
                    n_val=noise_val,
                    square_pos=self.square_pos,
                )
        return ret_data
    
    def get_square_mask_2D(self, dtype, device, data_shape, square_size, n_val, square_pos=None):
        img_h = data_shape[-2]
        img_w = data_shape[-1]

        img = torch.zeros(data_shape, dtype=dtype, device=device)

        if square_pos is None:
            w_start = np.random.randint(0, img_w - square_size)
            h_start = np.random.randint(0, img_h - square_size)
        else:
            pos_wh = square_pos[np.random.randint(0, len(square_pos))]
            w_start = pos_wh[0]
            h_start = pos_wh[1]

        rnd_n_val = self._get_range_val(n_val)
        img[:, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val

        return img
    
    def get_square_mask_3D(self, dtype, device, data_shape, square_size, n_val, square_pos=None):
        img_d = data_shape[-3]
        img_h = data_shape[-2]
        img_w = data_shape[-1]

        img = torch.zeros(data_shape, dtype=dtype, device=device)

        if square_pos is None:
            d_start = np.random.randint(0, img_d - square_size)
            w_start = np.random.randint(0, img_w - square_size)
            h_start = np.random.randint(0, img_h - square_size)
        else:
            pos_wh = square_pos[np.random.randint(0, len(square_pos))]
            d_start = pos_wh[0]
            w_start = pos_wh[1]
            h_start = pos_wh[2]

        rnd_n_val = self._get_range_val(n_val)
        img[:, d_start : (d_start + square_size), h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val

        return img
    
    def _get_range_val(self, value):
        if not isinstance(value, (list, tuple, np.ndarray)):
            return value
        if (
            len(value) == 2
            and value[0] == value[1]
            or len(value) != 2
            and len(value) == 1
        ):
            n_val = value[0]
        elif len(value) == 2:
            orig_type = type(value[0])
            if self.rnd_type == "uniform":
                n_val = random.uniform(value[0], value[1])
            elif self.rnd_type == "normal":
                n_val = random.normalvariate(value[0], value[1])
            n_val = orig_type(n_val)
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val

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