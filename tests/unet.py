#!/usr/bin/env python
  
import torch
from pytorch_unet import UNet
from pytorchviz import make_dot

class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        y = self.net(x)
        if isinstance(y, dict):
            y = tuple(y.values())
        return y


x = torch.rand(1, 1, 16, 16, 16).cuda().float()
unet = UNet(1, 4, 3, 8, output_levels=[3, 0, 1]).cuda()
unet = Wrapper(unet)
print(unet)
y = unet(x)

dot = make_dot(x, unet)
dot.format = 'pdf'
dot.render('unet')
