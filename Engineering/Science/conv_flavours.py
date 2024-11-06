#this will be moved to the Tricorder project

import math
import torch
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
from collections import abc as container_abcs 
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

################# DOConv ###########################
#Cite: Cao, Jinming, et al. "Do-conv: Depthwise over-parameterized convolutional layer." IEEE Transactions on Image Processing 31 (2022): 3726-3736.
#ArXiv: https://arxiv.org/abs/2006.12030
#Source doe: https://github.com/yangyanli/DO-Conv
####################################################

class DOConvNd(Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', n_dim=2):
        super(DOConvNd, self).__init__()
        
        self._pair = _ntuple(n_dim)
        kernel_size = self._pair(kernel_size)
        stride = self._pair(stride)
        padding = self._pair(padding)
        dilation = self._pair(dilation)
        
        match n_dim:
            case 1:
                self.conv_operator = F.conv1d
            case 2:
                self.conv_operator = F.conv2d
            case 3:
                self.conv_operator = F.conv3d

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return self.conv_operator(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            self._pair(0), self.dilation, self.groups)
        return self.conv_operator(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class DOConv1d(DOConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DOConv1d, self).__init__(in_channels, out_channels, kernel_size, D_mul, stride,
                                       padding, dilation, groups, bias, padding_mode, n_dim=1)

        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        self.D_mul = M if D_mul is None or M <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M > 1:
            self.D = Parameter(torch.Tensor(in_channels, M, self.D_mul))
            init_zero = np.zeros([in_channels, M, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M, dtype=torch.float32), (1, M, M))
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (M)))
            if self.D_mul % (M) != 0:  # the cases when D_mul > M
                zeros = torch.zeros([in_channels, M, self.D_mul % (M)])
                self.d_diag = Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M
                self.d_diag = Parameter(d_diag, requires_grad=False)
        ##################################################################################################

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        M = self.kernel_size[0]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M)
        if M > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M)
            D = self.D + self.d_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            # einsum outputs (out_channels // groups, in_channels, M),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            #######################################################
        else:
            # in this case D_mul == M
            # reshape from
            # (out_channels, in_channels // groups, D_mul)
            # to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(self.W, DoW_shape)
        return self._conv_forward(input, DoW)

class DOConv2d(DOConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DOConv2d, self).__init__(in_channels, out_channels, kernel_size, D_mul, stride,
                                       padding, dilation, groups, bias, padding_mode, n_dim=2)

        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.d_diag = Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.d_diag = Parameter(d_diag, requires_grad=False)
        ##################################################################################################

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N)
            D = self.D + self.d_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            #######################################################
        else:
            # in this case D_mul == M * N
            # reshape from
            # (out_channels, in_channels // groups, D_mul)
            # to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(self.W, DoW_shape)
        return self._conv_forward(input, DoW)
            
class DOConv3d(DOConvNd):    
    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DOConv3d, self).__init__(in_channels, out_channels, kernel_size, D_mul, stride,
                                       padding, dilation, groups, bias, padding_mode, n_dim=3)

        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        O = self.kernel_size[2]
        self.D_mul = M * N * O if D_mul is None or M * N * O <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N * O > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N * O, self.D_mul))
            init_zero = np.zeros([in_channels, M * N * O, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N * O, dtype=torch.float32), (1, M * N * O, M * N * O))
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N * O)))
            if self.D_mul % (M * N * O) != 0:  # the cases when D_mul > M * N * O
                zeros = torch.zeros([in_channels, M * N * O, self.D_mul % (M * N * O)])
                self.d_diag = Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N * O
                self.d_diag = Parameter(d_diag, requires_grad=False)
        ##################################################################################################

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        O = self.kernel_size[2]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N, O)
        if M * N * O > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N * O)
            D = self.D + self.d_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            # einsum outputs (out_channels // groups, in_channels, M * N * O),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N, O)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            #######################################################
        else:
            # in this case D_mul == M * N * O
            # reshape from
            # (out_channels, in_channels // groups, D_mul)
            # to
            # (out_channels, in_channels // groups, M, N, O)
            DoW = torch.reshape(self.W, DoW_shape)
        return self._conv_forward(input, DoW)
    
########### XSepConv ################
#Cite: Chen, Jiarong, Zongqing Lu, and Qingmin Liao. "XSepConv: extremely separated convolution for efficient deep networks with large kernels." Thirteenth International Conference on Digital Image Processing (ICDIP 2021). Vol. 11878. SPIE, 2021.
#ArXiv: https://arxiv.org/abs/2002.12046
#Code: NOPE
#####################################