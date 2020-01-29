#!/usr/bin/env python
# -*- coding: utf-8 -*-
  
import torch
from pytorch_unet import UNet
from pytorch_layers import Config, Dim
from torchviz import make_dot


Config.dim = Dim.THREE
Config.show()

unet = UNet(1, 4, 2, 8)
unet.output_levels = [0, 1, 3]
print(unet._output_levels)
# print(unet)

x = torch.rand(1, 1, 16, 16, 16)
y = unet(x)
print(y.shape)
dot = make_dot(y, params=dict(unet.named_parameters()))
dot.format = 'svg'
dot.render('unet')
