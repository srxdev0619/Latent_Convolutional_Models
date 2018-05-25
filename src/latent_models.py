from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F




class Latent4LSND(nn.Module):

    def __init__(self, lr_slope=0.2):
        super(Latent4LSND, self).__init__()
        self.lr_slope = lr_slope


        self.conv1 = nn.Conv2d(2, 8, 3, 1, 0, 1, bias=False) #80
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 0, 1, bias=False) #76
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 0, 1, bias=False) #72
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 0, 1, bias=False) #68

    def restrict(self, min_val, max_val):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    p.data.clamp_(min_val, max_val)

    def forward(self, x):

        x =  F.leaky_relu(self.conv1_bn(self.conv1(x)), self.lr_slope)
        x =  F.leaky_relu(self.conv2_bn(self.conv2(x)), self.lr_slope)
        x =  F.leaky_relu(self.conv3_bn(self.conv3(x)), self.lr_slope)
        x =  F.leaky_relu(self.conv4(x), self.lr_slope)

        return x
