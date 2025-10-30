import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import time
import scipy
import datetime
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Auxiliary functions
def smooth_series(series, window_size = 10):
    series_np = np.asarray(series)
    series_smooth = np.copy(series_np)

    overlap = int(window_size/2)

    for i in range(len(series_smooth)):
        series_smooth[i] = np.median(series_np[max(-overlap+i,0):min(overlap+i, len(series_smooth))])

    return series_smooth

def plot_loss(losses, model_id, smooth_window=100):
    plt.figure(figsize=(8,4))
    plt.plot(losses, alpha=0.5, label='raw loss')
    plt.plot(smooth_series(losses, smooth_window), label='smoothed loss')
    plt.title('Loss', fontweight='bold')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{model_id}/losses.png')
    plt.close()

# model blocks
class InProjection(nn.Module):
    def __init__(self, initial_channels=4):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(initial_channels, 10, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.in_proj(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val:float=0.1):
        super(DoubleConv, self).__init__()

        self.conv_net = nn.Sequential(
            self.conv_block(in_channels, out_channels, dropout_val),
            self.conv_block(out_channels, out_channels, dropout_val)
        )

    def forward(self, x):       
        return self.conv_net(x)
    
    def conv_block(self, in_channels, out_channels, dropout_val):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val, inplace=True),
            nn.LeakyReLU(inplace=True)
        )

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val=0.1):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, dropout_val)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class Up(nn.Module):
    def __init__(self, in_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

    def forward(self, x1):
        return self.up(x1)

# I got this AttentionBlock from https://github.com/sfczekalski/attention_unet/blob/master/att_unet.ipynb
class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

# the big boy
class AttUNet(nn.Module):
    def __init__(self, model_id, base_channels=64, dropout_val=0.1):
        super().__init__()

        self.model_id = model_id

        self.projection_xt = InProjection(4)
        self.projection_t = InProjection(1)

        self.enc_conv1 = DownSample(20, base_channels, dropout_val=dropout_val)
        self.enc_conv2 = DownSample(base_channels, base_channels*2, dropout_val=dropout_val)
        self.enc_conv3 = DownSample(base_channels*2, base_channels*4, dropout_val=dropout_val)
        self.enc_conv4 = DownSample(base_channels*4, base_channels*8, dropout_val=dropout_val)

        self.bottle_neck = DoubleConv(base_channels*8, base_channels*16, dropout_val=dropout_val)

        self.up1 = Up(base_channels*16)
        self.att1 = AttentionBlock(base_channels*8, base_channels*8, base_channels*8)
        self.dec_conv1 = DoubleConv(base_channels*16, base_channels*8, dropout_val=dropout_val)

        self.up2 = Up(base_channels*8)
        self.att2 = AttentionBlock(base_channels*4, base_channels*4, base_channels*4)
        self.dec_conv2 = DoubleConv(base_channels*8, base_channels*4, dropout_val=dropout_val)

        self.up3 = Up(base_channels*4)
        self.att3 = AttentionBlock(base_channels*2, base_channels*2, base_channels*2)
        self.dec_conv3 = DoubleConv(base_channels*4, base_channels*2, dropout_val=dropout_val)

        self.up4 = Up(base_channels*2)
        self.att4 = AttentionBlock(base_channels, base_channels, base_channels)
        self.dec_conv4 = DoubleConv(base_channels*2, base_channels, dropout_val=dropout_val)

        self.out = nn.Conv2d(in_channels=base_channels, out_channels=4, kernel_size=1)

    def forward(self, xt, t):
        x = torch.cat([self.projection_xt(xt), self.projection_t(t)], axis=1)
        
        down1, p1 = self.enc_conv1(x)
        down2, p2 = self.enc_conv2(p1)
        down3, p3 = self.enc_conv3(p2)
        down4, p4 = self.enc_conv4(p3)

        b = self.bottle_neck(p4)

        up1 = self.up1(b)
        att1 = self.att1(up1, down4)
        dec1 = self.dec_conv1(torch.concat([up1, att1], 1))

        up2 = self.up2(dec1)
        att2 = self.att2(up2, down3)
        dec2 = self.dec_conv2(torch.concat([up2, att2], 1))

        up3 = self.up3(dec2)
        att3 = self.att3(up3, down2)
        dec3 = self.dec_conv3(torch.concat([up3, att3], 1))

        up4 = self.up4(dec3)
        att4 = self.att4(up4, down1)
        dec4 = self.dec_conv4(torch.concat([up4, att4], 1))

        return self.out(dec4)