from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import zoom

import torch
import torch.nn as nn
from Engineering.Science.freq_trans import fftNc, ifftNc
from ..Science.complex import complex_modeconverter

##################Master Classes###########################


class SuperTransformer():
    def __init__(
            self,
            p: float = 1,
            include: Optional[Sequence[str]] = None,
            exclude: Optional[Sequence[str]] = None,
            # To skip all sample-level processes and all params, just simply call apply on the supplied tensor
            applyonly: bool = False,
            gt2inp: bool = False,
            **kwargs
    ):
        self.p = p
        self.include = [include] if type(include) is str else include
        self.exclude = [exclude] if type(exclude) is str else exclude
        self.applyonly = applyonly
        self.gt2inp = gt2inp
        self.return_meta = False

    def __call__(self, sample):
        if self.applyonly:
            out = self.apply(sample)
            if self.return_meta:
                return out[0]
            else:
                return out
        if self.gt2inp:
            if torch.rand(1).item() > self.p:
                sample['inp'] = deepcopy(sample['gt'])
            else:
                out = self.apply(sample['gt']['data'])
                if self.return_meta:
                    sample['inp'] = {
                    'data': out[0],
                    'path': ""
                    }
                    sample['inp'] = sample['inp'] | out[1]
                else:
                    sample['inp'] = {
                    'data': out,
                    'path': ""
                    }
                
        else:
            if torch.rand(1).item() > self.p:
                return sample
            for k in sample.keys():
                if (type(sample[k]) is not dict) or ("data" not in sample[k]) or (bool(self.include) and k not in self.include) or (not bool(self.include) and bool(self.exclude) and k in self.exclude):
                    continue
                if (isinstance(self, IntensityNorm) and "volmax" in sample[k]) or (isinstance(self, ZScoreNorm) and isinstance(self.pnorm, IntensityNorm) and "volmax" in sample[k]):
                    out = self.apply(sample[k]['data'], volminmax=(sample[k]["volmin"], sample[k]["volmax"]))
                else:
                    out = self.apply(sample[k]['data'])
                if self.return_meta:
                    sample[k]['data'] = out[0]
                    sample[k] = sample[k] | out[1]
                else:
                    sample[k]['data'] = out
        return sample


class ApplyOneOf():
    def __init__(
            self,
            transforms_dict
    ):
        self.transforms_dict = transforms_dict

    def __call__(self, inp):
        weights = torch.Tensor(list(self.transforms_dict.values()))
        index = torch.multinomial(weights, 1)
        transforms = list(self.transforms_dict.keys())
        transform = transforms[index]
        return transform(inp)

###########################################################

################Transformation Functions###################


def padIfNeeded(inp, size=None):
    inp_shape = inp.shape 
    size = size if len(inp.shape) == len(size) else (inp.shape[0], *size)
    pad = [(0, 0), ]*len(inp_shape)
    pad_requried = False
    for i in range(len(inp_shape)):
        if inp_shape[i] < size[i]:
            diff = size[i]-inp_shape[i]
            pad[i] = (diff//2, diff-(diff//2))
            pad_requried = True
    if not pad_requried:
        return inp
    else:
        return np.pad(inp, pad)


def cropcentreIfNeeded(inp, size=None):
    if len(inp.shape) == len(size):
        if len(inp.shape) == 2:
            x, y = inp.shape
        else:
            x, y, z = inp.shape
        if bool(size[0]) and x > size[0]:
            diff = x-size[0]
            inp = inp[diff//2:diff//2+size[0], ...]
        if bool(size[1]) and y > size[1]:
            diff = y-size[1]
            if len(inp.shape) == 2:
                inp = inp[..., diff//2:diff//2+size[1]]
            else:
                inp = inp[:, diff//2:diff//2+size[1], :]
        if len(inp.shape) == 3 and z > size[2]:
            diff = z-size[2]
            inp = inp[..., diff//2:diff//2+size[2]]
    else:        
        if len(inp.shape) == 3:
            _, x, y = inp.shape
        else:
            _, x, y, z = inp.shape
        if bool(size[0]) and x > size[0]:
            diff = x-size[0]
            inp = inp[:, diff//2:diff//2+size[0], ...]
        if bool(size[1]) and y > size[1]:
            diff = y-size[1]
            if len(inp.shape) == 3:
                inp = inp[..., diff//2:diff//2+size[1]]
            else:
                inp = inp[:, :, diff//2:diff//2+size[1], :]
        if len(inp.shape) == 4 and z > size[2]:
            diff = z-size[2]
            inp = inp[..., diff//2:diff//2+size[2]]
    return inp

class Interpolate(SuperTransformer):
    def __init__(
            self,
            factor: int = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.factor = factor

    def apply(self, inp):
        return zoom(inp, self.factor, order=1)

class CropOrPad(SuperTransformer):
    def __init__(
            self,
            size: Union[Tuple[int], str],
            **kwargs
    ):
        super().__init__(**kwargs)
        if type(size) == str:
            size = tuple([int(tmp) for tmp in size.split(",")])
        self.size = size

    def apply(self, inp):
        inp = padIfNeeded(inp, size=self.size)
        return cropcentreIfNeeded(inp, size=self.size)

class IntensityNorm(SuperTransformer):
    def __init__(
            self,
            type: str = "minmax", 
            return_meta: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.type = type
        self.return_meta = return_meta

    def apply(self, inp, volminmax=None):
        if volminmax is None:
            vmin = inp.min()
            vmax = inp.max()
        else:
            vmin, vmax = volminmax
        if "minmax" in self.type:
            if self.return_meta:
                return (inp - vmin) / (vmax - vmin + np.finfo(np.float32).eps), {"NormMeta": {"min": vmin, "max": vmax}}
            else:
                return (inp - vmin) / (vmax - vmin + np.finfo(np.float32).eps)
        elif "divbymax" in self.type:
            if self.return_meta:
                return inp / (vmax + np.finfo(np.float32).eps), {"NormMeta": {"max": vmax}}
            else:
                return inp / (vmax + np.finfo(np.float32).eps)

class ZScoreNorm(SuperTransformer):
    def __init__(
            self,
            dim: int = 2,
            mean: Optional[Union[None, tuple]] = None,
            std: Optional[Union[None, tuple]] = None,
            prenorm: Optional[Union[None, str]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.mean = mean
        self.std = std
        if bool(prenorm):
            try:
                prenorm = float(prenorm)
                self.pnorm = lambda inp, _: inp / prenorm 
            except Exception: #If prenorm is not a float, it must be a type of IntensityNorm
                self.prenorm_type = prenorm
        else:
            self.pnorm = nn.Identity()
            
    def pnorm(self, inp, volminmax=None):
        if volminmax is None:
            vmin = inp.min()
            vmax = inp.max()
        else:
            vmin, vmax = volminmax
        if "minmax" in self.prenorm_type:
            return (inp - vmin) / (vmax - vmin + np.finfo(np.float32).eps)
        elif "divbymax" in self.prenorm_type:
            return inp / (vmax + np.finfo(np.float32).eps)

    def apply(self, inp, mean=None, std=None, volminmax=None):
        assert type(inp) is torch.Tensor, "Input to the ZScoreNorm's apply function must be a torch tensor"
        inp = self.pnorm(inp, volminmax=volminmax)
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std
        mean = torch.as_tensor(mean, dtype=inp.dtype, device=inp.device)
        std = torch.as_tensor(std, dtype=inp.dtype, device=inp.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {inp.dtype}, leading to division by zero.")
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1) if self.dim==2 else mean.view(-1, 1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1) if self.dim==2 else std.view(-1, 1, 1, 1)
        return inp.sub_(mean).div_(std)

class ToTensor(SuperTransformer):
    def __init__(
            self,
            dim: int = 2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim

    def apply(self, inp):
        if (self.dim == 2 and len(inp.shape) == 2) or (self.dim == 3 and len(inp.shape) == 3):
            inp = inp[np.newaxis, ...]
        return torch.from_numpy(inp)
    
class Grey2RGB(SuperTransformer):
    def __init__(
            self,
            mode: int = 1,
            ch_dim: int = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.ch_dim = ch_dim

    def apply(self, inp):
        if self.mode == 0:
            return torch.concat([torch.zeros_like(inp), inp, torch.zeros_like(inp)], axis=self.ch_dim)
        elif self.mode == 1:
            return torch.concat([inp, inp, inp], axis=self.ch_dim)
        elif self.mode == 2:
            #torch.bfloat16 was chosen to avoid any kind of underflow or overflow
            return torch.concat([torch.rand_like(inp) * (2*torch.finfo(torch.bfloat16).eps) - torch.finfo(torch.bfloat16).eps, 
                                 inp, 
                                 torch.rand_like(inp) * (2*torch.finfo(torch.bfloat16).eps) - torch.finfo(torch.bfloat16).eps], axis=self.ch_dim)
        else:
            raise ValueError("ch_dim must be 0 or 1 for mode 1")
        
        
class ComplexModeConverter(SuperTransformer):
    def __init__(
            self,
            complex_mode: int = 0,
            channel_dim_present: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.complex_mode = complex_mode
        self.channel_dim_present = channel_dim_present

    def apply(self, inp):
        return complex_modeconverter(inp, self.complex_mode, self.channel_dim_present)
    
class ChangeDataSpace(SuperTransformer):
    def __init__(
            self,
            source_data_space,
            destin_data_space,
            data_dim=(-3, -2, -1),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.source_data_space = source_data_space
        self.destin_data_space = destin_data_space
        self.data_dim = data_dim

    def apply(self, inp):
        if self.source_data_space == 0 and self.destin_data_space == 1:
            return fftNc(inp, dim=self.data_dim)
        elif self.source_data_space == 1 and self.destin_data_space == 0:
            return ifftNc(inp, dim=self.data_dim)


def getDataSpaceTransforms(dataspace_inp, model_dataspace_inp, dataspace_gt, model_dataspace_gt):
    if dataspace_inp == dataspace_gt and model_dataspace_inp == model_dataspace_gt and dataspace_inp != model_dataspace_inp:
        return [ChangeDataSpace(dataspace_inp, model_dataspace_inp)]
    else:
        trans = []
        if dataspace_inp != model_dataspace_inp and dataspace_inp != -1 and model_dataspace_inp != -1:
            trans.append(ChangeDataSpace(
                dataspace_inp, model_dataspace_inp, include="inp"))
        elif dataspace_gt != model_dataspace_gt and dataspace_gt != -1 and model_dataspace_gt != -1:
            trans.append(ChangeDataSpace(
                dataspace_gt, model_dataspace_gt, include="gt"))
        return trans

###########################################################
