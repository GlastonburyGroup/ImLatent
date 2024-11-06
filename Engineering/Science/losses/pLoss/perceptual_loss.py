import math
import sys
import logging

import torch
import torch.nn as nn
import torchvision
# from utils.utils import *
from pytorch_msssim import MS_SSIM, SSIM

from .Resnet2D import ResNet
from .simpleunet import UNet
from .VesselSeg_UNet3d_DeepSup import U_Net_DeepSup


# currently configured for 1 channel only, with datarange as 1 for SSIM
class PerceptualLoss(torch.nn.Module):
    def __init__(self, loss_model="med1ch3D.UNetMSSDS6", n_level=math.inf, resize=None, loss_type="L1", mean=[], std=[], n_channels=1, n_dim=3):
        super(PerceptualLoss, self).__init__()
        blocks = []

        if "tv" in loss_type.lower():
            loss_model_parts = loss_model.split(".")
            model = torch.hub.load("pytorch/vision", loss_model_parts[1], weights=loss_model_parts[2])
            #non-RestNet models
            if "features" in model._modules.keys():
                n_block_splits = int(loss_model_parts[3])
                n_features_each_block = len(model.features) // n_block_splits
                for i in range(n_block_splits):
                    blocks.append(model.features[i*n_features_each_block:(i+1)*n_features_each_block].eval())
            #ResNet models
            elif "layer1" in model._modules.keys() and "layer2" in model._modules.keys() and "layer3" in model._modules.keys() and "layer4" in model._modules.keys(): 
                blocks.append(nn.Sequential(model.conv1.eval(),
                                            model.bn1.eval(),
                                            model.relu.eval(),
                                            model.maxpool.eval()))
                blocks.append(model.layer1.eval())
                blocks.append(model.layer2.eval())
                blocks.append(model.layer3.eval())
                blocks.append(model.layer4.eval())
            else:
                sys.exit("Perceptual Loss: TV without either features or layer1-4 not implemented")
        elif loss_model == "med1ch2D.ResNet":  # TODO: not finished
            sys.exit("Perceptual Loss: resnet2D PLN not finished")
            model = ResNet(in_channels=1, out_channels=1)
            chk = torch.load(
                r"./Engineering/Science/losses/pLoss/pretrained_weights_1ch/ResNet14_IXIT2_Base_d1p75_t0_n10_dir01_5depth_L1Loss_best.pth.tar", map_location="cpu")
            model.load_state_dict(chk['state_dict'])
        elif loss_model == "med1ch2D.UNet":
            model = UNet(in_channels=1, out_channels=1, depth=5, wf=6, padding=True,
                         batch_norm=False, up_mode='upsample', droprate=0.0, is3D=False,
                         returnBlocks=False, downPath=True, upPath=True)
            chk = torch.load(
                r"./Engineering/Science/losses/pLoss/pretrained_weights_1ch/SimpleU_IXIT2_Base_d1p75_t0_n10_dir01_5depth_L1Loss_best.pth.tar", map_location="cpu")
            model.load_state_dict(chk['state_dict'])
            blocks.append(model.down_path[0].block.eval())
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(2),
                        model.down_path[1].block.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(2),
                        model.down_path[2].block.eval()
                    )
                )
            if n_level >= 4:
                blocks.append(
                    nn.Sequential(
                        nn.AvgPool2d(2),
                        model.down_path[3].block.eval()
                    )
                )
        elif loss_model == "med1ch3D.UNetMSSDS6":
            model = U_Net_DeepSup()
            chk = torch.load(
                r"./Engineering/Science/losses/pLoss/pretrained_weights_1ch/VesselSeg_UNet3d_DeepSup.pth", map_location="cpu")
            model.load_state_dict(chk['state_dict'])
            blocks.append(model.Conv1.conv.eval())
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool1.eval(),
                        model.Conv2.conv.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool2.eval(),
                        model.Conv3.conv.eval()
                    )
                )
            if n_level >= 4:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool3.eval(),
                        model.Conv4.conv.eval()
                    )
                )
            if n_level >= 5:
                blocks.append(
                    nn.Sequential(
                        model.Maxpool4.eval(),
                        model.Conv5.conv.eval()
                    )
                )
        elif loss_model == "med1ch2D.ResNeXt101":
            model = torchvision.models.resnext101_32x8d()
            model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                                    stride=model.conv1.stride, padding=model.conv1.padding, bias=False if model.conv1.bias is None else True)
            model.fc = nn.Linear(in_features=model.fc.in_features,
                                 out_features=33, bias=False if model.fc.bias is None else True)
            chk = torch.load(r"./Engineering/Science/losses/pLoss/pretrained_weights_1ch/ResNeXt-3-class-best-latest.pth", map_location="cpu")
            model.load_state_dict(chk)
            blocks.append(
                nn.Sequential(
                    model.conv1.eval(),
                    model.bn1.eval(),
                    model.relu.eval(),
                )
            )
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        model.maxpool.eval(),
                        model.layer1.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(model.layer2.eval())
            if n_level >= 4:
                blocks.append(model.layer3.eval())
            if n_level >= 5:
                blocks.append(model.layer4.eval())
        elif loss_model == "med1ch2D.DenseNet161":
            logging.critical("Perceptual Loss: Weights for DenseNet161 as PLN not available")
            sys.exit("Perceptual Loss: Weights for DenseNet151 as P6N not available")
            model = torchvision.models.densenet161()
            model.features.conv0 = nn.Conv2d(1, model.features.conv0.out_channels, kernel_size=model.features.conv0.kernel_size,
                                             stride=model.features.conv0.stride, padding=model.features.conv0.padding,
                                             bias=False if model.features.conv0.bias is None else True)
            model.classifier = nn.Linear(in_features=model.classifier.in_features,
                                         out_features=33, bias=False if model.classifier.bias is None else True)
            chk = torch.load(r"./Engineering/Science/losses/pLoss/pretrained_weights_1ch/ResNet14_IXIT2_Base_d1p75_t0_n10_dir01_5depth_L1Loss_best.pth.tar", map_location="cpu")
            model.load_state_dict(chk['state_dict'])
            model = model.features
            blocks.append(
                nn.Sequential(
                    model.conv0.eval(),
                    model.norm0.eval(),
                    model.relu0.eval(),
                )
            )
            if n_level >= 2:
                blocks.append(
                    nn.Sequential(
                        model.pool0.eval(),
                        model.denseblock1.eval()
                    )
                )
            if n_level >= 3:
                blocks.append(model.denseblock2.eval())
            if n_level >= 4:
                blocks.append(model.denseblock3.eval())
            if n_level >= 5:
                blocks.append(model.denseblock4.eval())

        for bl in blocks:
            for params in bl.parameters():
                params.requires_grad = False

        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        if (mean is not None and len(mean) > 1) and (std is not None and len(std) > 1) and (len(mean) == len(std)):
            self.mean = nn.Parameter(
                torch.tensor(mean).view(1, len(mean), 1, 1))
            self.std = nn.Parameter(torch.tensor(std).view(1, len(std), 1, 1))
        else:
            self.mean = None
            self.std = None
        self.resize = resize

        if loss_type == "L1":
            self.loss_func = torch.nn.functional.l1_loss
        elif loss_type == "MultiSSIM":
            self.loss_func = MS_SSIM(reduction='mean')
        elif loss_type == "SSIM3D":
            self.loss_func = SSIM(
                data_range=1, size_average=True, channel=1, spatial_dims=3)
        elif loss_type == "SSIM2D":
            self.loss_func = SSIM(
                data_range=1, size_average=True, channel=1, spatial_dims=2)

    def forward(self, input, target):
        if self.mean is not None:
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='trilinear' if len(
                input.shape) == 5 else 'bilinear', size=self.resize, align_corners=False)
            target = self.transform(target, mode='trilinear' if len(
                input.shape) == 5 else 'bilinear', size=self.resize, align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += self.loss_func(x, y)
        return loss


if __name__ == '__main__':
    x = PerceptualLoss(resize=None).cuda()
    a = torch.rand(2, 1, 24, 24).cuda()
    b = torch.rand(2, 1, 24, 24).cuda()
    l = x(a, b)
    print(l)
