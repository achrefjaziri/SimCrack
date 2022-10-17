"""Implement the UNet architecture for segmentation
source: https://github.com/milesial/Pytorch-UNet

"""
import os
import sys
a_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,a_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from lib.models.conv_layers import inconv,double_conv,down,up,outconv


class MultiUNet(nn.Module):
    """Initalizes the UNet architecture and the according forward pass"""
    def __init__(self, n_channels, n_classes):
        """Initialize a U-Net object
        Set the third variable in the up instance = False if feature maps shouldn't
        be upsampled by bilinear interpolation but by learning the weights"""
        super(MultiUNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.up1D = up(1024, 256)
        self.up2D = up(512, 128)
        self.up3D = up(256, 64)
        self.up4D = up(128, 64)
        self.up1N = up(1024, 256)
        self.up2N = up(512, 128)
        self.up3N = up(256, 64)
        self.up4N = up(128, 64)
        self.outdepth = outconv(64, 1)
        self.outnormals = outconv(64, 3)
        self.outc = outconv(64,2)

    def forward(self, x):
        """Define the forward pass"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_seg = self.up1(x5, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x1)
        x_depth = self.up1D(x5, x4)
        x_depth = self.up2D(x_depth, x3)
        x_depth = self.up3D(x_depth, x2)
        x_depth = self.up4D(x_depth, x1)
        x_norm = self.up1N(x5, x4)
        x_norm = self.up2N(x_norm, x3)
        x_norm = self.up3N(x_norm, x2)
        x_norm = self.up4N(x_norm, x1)
        x_norm = self.outnormals(x_norm)
        x_seg = self.outc(x_seg)
        x_depth = self.outdepth(x_depth)

        return x_norm ,x_seg,x_depth

class MultiUNetEncoder(nn.Module):
    """Initalizes the UNet architecture and the according forward pass"""
    def __init__(self, n_channels, n_classes):
        """Initialize a U-Net object
        Set the third variable in the up instance = False if feature maps shouldn't
        be upsampled by bilinear interpolation but by learning the weights"""
        super(MultiUNetEncoder, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)


    def forward(self, x):
        """Define the forward pass"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return {'x1':x1,'x2':x2,'x3':x3,'x4':x4,'x5':x5}





class MultiUNetDecoder(nn.Module):
    """Initalizes the UNet architecture and the according forward pass"""

    def __init__(self, n_channels, n_classes):
        """Initialize a U-Net object
        Set the third variable in the up instance = False if feature maps shouldn't
        be upsampled by bilinear interpolation but by learning the weights"""
        super(MultiUNetEncoder, self).__init__()
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.up1D = up(1024, 256)
        self.up2D = up(512, 128)
        self.up3D = up(256, 64)
        self.up4D = up(128, 64)
        self.up1N = up(1024, 256)
        self.up2N = up(512, 128)
        self.up3N = up(256, 64)
        self.up4N = up(128, 64)
        self.outdepth = outconv(64, 1)
        self.outnormals = outconv(64, 3)
        self.outc = outconv(64, 2)

    def forward(self, x):
        """Define the forward pass"""


        x_seg = self.up1(x['x5'], x['x4'])
        x_seg = self.up2(x_seg, x['x3'])
        x_seg = self.up3(x_seg,x['x2'] )
        x_seg = self.up4(x_seg, x['x1'])
        x_depth = self.up1D(x['x5'], x['x4'])
        x_depth = self.up2D(x_depth, x['x3'])
        x_depth = self.up3D(x_depth, x['x2'])
        x_depth = self.up4D(x_depth, x['x1'])
        x_norm = self.up1N(x['x5'], x['x4'])
        x_norm = self.up2N(x_norm, x['x3'])
        x_norm = self.up3N(x_norm, x['x2'])
        x_norm = self.up4N(x_norm, x['x1'])
        x_norm = self.outnormals(x_norm)
        x_seg = self.outc(x_seg)
        x_depth = self.outdepth(x_depth)

        return {'nrm':x_norm, 'seg':x_seg, 'depth': x_depth}
if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(2, 1, 512, 512)
    model = MultiUNet(1, 2)
    summary(model, input_size=(2, 1, 512, 512))
    #print(x.shape)