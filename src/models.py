from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import torch
import torch.nn as nn
import torch.nn.functional as F







class EncDecCelebA(nn.Module):

    def __init__(self, in_channels=1, lr_slope=0.2, bias=False):
        super(EncDecCelebA, self).__init__()
        self.lr_slope = lr_slope

        self.conv1 = nn.Conv2d(in_channels, 256, 4, 2, 1, 1, bias=False) #16
        self.conv1_bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, 4, 2, 1, 1, bias=False) #8
        self.conv2_bn = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, 4, 2, 1, 1, bias=False) #4
        self.conv3_bn = nn.BatchNorm2d(1024)

        self.conv4 = nn.Conv2d(1024, 1024, 3, 1, 2, 2, groups=512, bias=False) #4
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1024, 3, 1, 2, 2, groups=512, bias=False) #4
        self.conv5_bn = nn.BatchNorm2d(1024)
        self.conv6 = nn.Conv2d(1024, 1024, 3, 1, 2, 2, groups=512, bias=False) #4
        self.conv6_bn = nn.BatchNorm2d(1024)

        self.convT1 = nn.ConvTranspose2d(1024 + 1024, 512, 4, 2, 1, bias=bias)
        self.convT1_bn = nn.BatchNorm2d(512) #8
        self.convT2 = nn.ConvTranspose2d(512+512, 512, 4, 2, 1,  bias=bias)
        self.convT2_bn = nn.BatchNorm2d(512) #16
        self.convT3 = nn.ConvTranspose2d(512, 256, 4, 2, 1,  bias=bias)
        self.convT3_bn = nn.BatchNorm2d(256) #32
        self.convT4 = nn.ConvTranspose2d(256, 128, 4, 2, 1,  bias=bias)
        self.convT4_bn = nn.BatchNorm2d(128) #64
        self.convT5 = nn.Conv2d(128, 64, 3, 1, 1,  bias=bias)
        self.convT5_bn = nn.BatchNorm2d(64) #128
        self.convT6 = nn.Conv2d(64, 32, 3, 1, 1, 1, bias=bias)
        self.convT6_bn = nn.BatchNorm2d(32) #128 ##
        self.convT7 = nn.Conv2d(32, 3, 3, 1, 1, 1, bias=bias) ##
        self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, input):
        #Encoder
        x1 =  F.leaky_relu(self.conv1_bn(self.conv1(input)), self.lr_slope)
        x2 =  F.leaky_relu(self.conv2_bn(self.conv2(x1)), self.lr_slope)
        x3 =  F.leaky_relu(self.conv3_bn(self.conv3(x2)), self.lr_slope)

        x4 =  F.leaky_relu(self.conv4_bn(self.conv4(x3)), self.lr_slope)
        x5 =  F.leaky_relu(self.conv5_bn(self.conv5(x4)), self.lr_slope)
        x6 =  F.leaky_relu(self.conv6_bn(self.conv6(x5)), self.lr_slope)


        #Decoder
        x = torch.cat([x6,x3],1)
        x =  F.leaky_relu(self.convT1_bn(self.convT1(x)), self.lr_slope) #8
        x = torch.cat([x,x2], 1)
        x =  F.leaky_relu(self.convT2_bn(self.convT2(x)), self.lr_slope) #16
        x =  F.leaky_relu(self.convT3_bn(self.convT3(x)), self.lr_slope) #32
        x =  F.leaky_relu(self.convT4_bn(self.convT4(x)), self.lr_slope) #64
        x = self.upsamp(x)
        x =  F.leaky_relu(self.convT5_bn(self.convT5(x)), self.lr_slope) #128
        x =  F.leaky_relu(self.convT6_bn(self.convT6(x)), self.lr_slope) #128
        x =  F.sigmoid(self.convT7(x)) #128

        return x
