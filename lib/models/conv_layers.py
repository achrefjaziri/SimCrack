"""Implements common conv layers used in Unet and MultiUnet
source: https://github.com/milesial/Pytorch-UNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    """"apply a convolution layer followed by batch normalization and a ReLU function twice"""
    def __init__(self, in_channels, out_channels,dropout=False):
        super(double_conv, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(p=0.5))
        self.conv = nn.Sequential(*layers)

        #self.conv = nn.Sequential(
        #    nn.Conv2d(in_channels, out_channels, 3, padding=1),
        #    nn.BatchNorm2d(out_channels),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #    nn.BatchNorm2d(out_channels),
        #    nn.ReLU(inplace=True)
            #add droupout here
        #)

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    """Downsamples the input using a 2x2 max pooling kernel and applies
    double convolution. This step reduces the resolution of the input and
    increases the amount of feature maps to out_channels"""
    def __init__(self, in_channels, out_channels, dropout =False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, out_channels,dropout)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class up(nn.Module):
    """Upsamples the input using either by applying bilinear upsampling or by learning
    the weights for upsampling
    Set bilinear = False if weights should learn to upsample
    """
    def __init__(self, in_channels, out_channels, bilinear=False,dropout =False,attention_connections=False):
        super(up, self).__init__()
        # next feature: learn to upsample
        self.att_con = attention_connections
        if bilinear:
            # upsample by learning bilinear interpolation
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # upsample by learning the respective weights
            # the upconvolutions preserves the number of input channels coming from the previous
            # step in the expansion path
            # the number of channels gets then reduced by the double convolution
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

        if attention_connections:
            self.att = Attention_block(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)

        # apply the double convolution to the weights
        self.conv = double_conv(in_channels, out_channels,dropout)

    def forward(self, x1, x2):
        if self.att_con:
            x = self.up(x1)
            x2 = self.att(g=x, x=x2)
            x = torch.cat((x, x2), dim=1)
            x = self.conv(x)
            return x
        else:
            x1 = self.up(x1)
            # input consists of color x height x width
            # x1 is the upsampled tensor, x2 is respective tensor from the contraction path
            height_difference = x2.size()[2] - x1.size()[2]
            width_difference = x2.size()[3] - x1.size()[3]

            # pads the tensor x1 so that tensor x2 and x1 have equal dimensions and can be concatenated
            x1 = F.pad(x1, (width_difference // 2, width_difference - width_difference // 2,
                            height_difference // 2, height_difference - height_difference // 2))

            # concatenates tensors x2 and x1 for Dimsension 1 (channels)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x






class outconv(nn.Module):
    '''Passes the output of the forelast layer through a convolutional layer to
    generate the networks output'''
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        return x