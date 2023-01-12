import os
import sys

a_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, a_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from lib.models.conv_layers import inconv, double_conv, down, up, outconv


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

def _regression_loss(x, y):
    # eps = 1e-6 if torch.is_autocast_enabled() else 1e-12
    x = F.normalize(x, p=2, dim=1)  # , eps=eps)
    y = F.normalize(y, p=2, dim=1)  # , eps=eps)
    return (2 - 2 * (x * y).sum(dim=1)).view(-1)



class ConsNet(nn.Module):
    """Initalizes the UNet architecture and the according forward pass"""

    def __init__(self, n_channels, n_classes,att=False,consistent_features=False,img_size=256):
        """Initialize a U-Net object
        Set the third variable in the up instance = False if feature maps shouldn't
        be upsampled by bilinear interpolation but by learning the weights"""
        super(ConsNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256,attention_connections=att)
        self.up2 = up(512, 128,attention_connections=att)
        self.up3 = up(256, 64,attention_connections=att)
        self.up4 = up(128, 64,attention_connections=att)

        self.incPMI = inconv(n_channels, 64)
        self.down1PMI = down(64, 128)
        self.down2PMI = down(128, 256)
        self.down3PMI = down(256, 512)
        self.down4PMI = down(512, 512)
        self.up1PMI = up(1024, 256,attention_connections=att)
        self.up2PMI = up(512, 128,attention_connections=att)
        self.up3PMI = up(256, 64,attention_connections=att)
        self.up4PMI = up(128, 64,attention_connections=att)

        self.outc = outconv(64, n_classes)
        self.outcPMI = outconv(64, n_classes)

        self.att = att
        self.consistent_features = consistent_features

        if consistent_features:
            self.project_head_rgb = nn.Sequential(
                nn.Linear(512*(img_size//16)**2, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 64, bias=False)
            )
            self.project_head_pmi = nn.Sequential(
                nn.Linear(512*(img_size//16)**2, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 64, bias=False)
            )

    def forward(self, input_rgb,input_pmi):
        x1 = self.inc(input_rgb)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        features_rgb = self.down4(x4)

        x_rgb = self.up1(features_rgb, x4)
        x_rgb = self.up2(x_rgb, x3)
        x_rgb = self.up3(x_rgb, x2)
        x_rgb = self.up4(x_rgb, x1)
        x_rgb = self.outc(x_rgb)

        x1 = self.incPMI(input_pmi)
        x2 = self.down1PMI(x1)
        x3 = self.down2PMI(x2)
        x4 = self.down3PMI(x3)
        features_pmi = self.down4PMI(x4)
        x_pmi = self.up1PMI(features_pmi, x4)
        x_pmi = self.up2PMI(x_pmi, x3)
        x_pmi = self.up3PMI(x_pmi, x2)
        x_pmi = self.up4PMI(x_pmi, x1)
        x_pmi = self.outcPMI(x_pmi)
        consistency_loss=input_rgb #This is just a placeholder.
        if self.consistent_features:
            flattened_features_rgb = features_rgb.reshape(features_rgb.shape[0], -1)
            flattened_features_pmi = features_pmi.reshape(features_pmi.shape[0], -1)
            projection_pmi = self.project_head_pmi(flattened_features_pmi)
            projection_rgb = self.project_head_rgb(flattened_features_rgb)
            consistency_loss = _regression_loss(projection_pmi,projection_rgb).mean()
        return x_rgb,x_pmi,consistency_loss

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(2, 1, 256, 256)
    model = ConsNet(1, 2,att=True,consistent_features=True)
    outs,outs2,loss = model(im,im)
    #summary(model, input_size=(2, 1, 256, 256))
    print(outs.shape,outs2.shape,loss)


    #TODO fix the sum part of x_rgb and x_pmi
    #TODO start training one with nothing and one with attention
    #After I get results for Attention single/double and double. Check for the impact of Consistency and the impact of Style Transfer