"""Implement the UNet architecture for segmentation
source: https://github.com/milesial/Pytorch-UNet

"""
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
a_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,a_path)
from lib.models.conv_layers import inconv,double_conv,down,up,outconv


class UNet(nn.Module):
    """Initalizes the UNet architecture and the according forward pass"""
    def __init__(self, n_channels, n_classes):
        """Initialize a U-Net object
        Set the third variable in the up instance = False if feature maps shouldn't
        be upsampled by bilinear interpolation but by learning the weights"""
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        """Define the forward pass"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UNet_Encoder(nn.Module):
    """UNet Encoder architecture"""
    def __init__(self, n_channels):
        """Initialize a U-Net object
        Set the third variable in the up instance = False if feature maps shouldn't
        be upsampled by bilinear interpolation but by learning the weights"""
        super(UNet_Encoder, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        #self.up1 = up(1024, 256)
        #self.up2 = up(512, 128)
        #self.up3 = up(256, 64)
        #self.up4 = up(128, 64)
        #self.outc = outconv(64, n_classes)

    def forward(self, x):
        """Define the forward pass"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x = self.up2(x, x3)
        #x = self.up3(x, x2)
        #x = self.up4(x, x1)
        #x = self.outc(x)
        return {'x1':x1,'x2':x2,'x3':x3,'x4':x4,'x5':x5}


class UNet_Decoder(nn.Module):
    """ UNet Decoder architecture"""
    def __init__(self,  n_classes):
        """Initialize a U-Net object
        Set the third variable in the up instance = False if feature maps shouldn't
        be upsampled by bilinear interpolation but by learning the weights"""
        super(UNet_Encoder, self).__init__()
        #self.inc = inconv(n_channels, 64)
        #self.down1 = down(64, 128)
        #self.down2 = down(128, 256)
        #self.down3 = down(256, 512)
        #self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, encoding):
        """Define the forward pass"""
        #x1 = self.inc(x)
        #x2 = self.down1(x1)
        #x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        x = self.up1(encoding['x5'],encoding['x4'])
        x = self.up2(x, encoding['x3'])
        x = self.up3(x, encoding['x2'])
        x = self.up4(x, encoding['x1'])
        x = self.outc(x)
        return x


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(4, 1, 512, 512)
    model = UNet(1, 2)
    x = model(im)
    print(im.shape)
    print(x.shape)
    summary(model, input_size=(2, 1, 512, 512))
