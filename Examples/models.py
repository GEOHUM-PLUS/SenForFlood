import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import time
import scipy
import datetime
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class AttentionGate(nn.Module):
    def __init__(self, g_channels, s_channels, out_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, g, s):
        g1 = self.Wg(g)          # Decoder features
        s1 = self.Ws(s)          # Skip connection features
        out = self.relu(g1 + s1) # Merge signals
        psi = self.psi(out)      # Attention map (0 to 1)
        return s * psi           # Filtered skip

class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, dropout:float=0.2):
        super().__init__()
        self.t_emb = nn.Conv2d(1, out_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.drop1 = nn.Dropout(dropout, inplace=True)
        self.selu1 = nn.SELU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop2 = nn.Dropout(dropout, inplace=True)
        self.selu2 = nn.SELU(inplace=True)

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.selu1(x)

        te = t[:, None, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        te = self.t_emb(te)

        x = x+te

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.selu2(x)

        return x

class AttUNet_t(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base=64, dropout=0.2, use_terrain=False):
        super().__init__()
        self.base = base
        self.use_terrain = use_terrain
        self.dropout = dropout

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transp_conv1 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.transp_conv2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.transp_conv3 = nn.ConvTranspose2d(base*2, base*1, kernel_size=2, stride=2)
        
        self.conv1 = ConvBlock(
            in_channels=in_channels+1 if use_terrain else in_channels, 
            out_channels=base,
            dropout=dropout
        )
        
        self.conv2 = ConvBlock(
            in_channels=base, 
            out_channels=base*2,
            dropout=dropout
        )
        
        self.conv3 = ConvBlock(
            in_channels=base*2, 
            out_channels=base*4,
            dropout=dropout
        )
        
        self.conv4 = ConvBlock(
            in_channels=base*4, 
            out_channels=base*8,
            dropout=dropout
        )
        
        self.conv5 = ConvBlock(
            in_channels=base*4+base*4, 
            out_channels=base*4,
            dropout=dropout
        )
        
        self.conv6 = ConvBlock(
            in_channels=base*2+base*2, 
            out_channels=base*2,
            dropout=dropout
        )
        
        self.conv7 = ConvBlock(
            in_channels=base+base, 
            out_channels=base,
            dropout=dropout
        )

        self.att1 = AttentionGate(base*4, base*4, base*4)
        self.att2 = AttentionGate(base*2, base*2, base*2)
        self.att3 = AttentionGate(base, base, base)

        self.out = nn.Conv2d(in_channels=base, out_channels=out_channels, kernel_size=1, padding=0, padding_mode='reflect')

    def get_conv_block(self, in_channels, out_channels, dropout_val):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val, inplace=True),
            nn.SELU(inplace=True)
        )
    
    def get_in_proj(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            # nn.BatchNorm2d(16),
            nn.SELU(inplace=True)
        )

    def forward(self, xt, t, xterr=None):
        if self.use_terrain:
            xt = torch.cat([xt,xterr], axis=1)

        end1 = self.conv1(xt,t)
        down1 = self.pool(end1)

        end2 = self.conv2(down1,t)
        down2 = self.pool(end2)

        end3 = self.conv3(down2,t)
        down3 = self.pool(end3)

        end4 = self.conv4(down3,t)
        up1 = self.transp_conv1(end4)

        a1 = self.att1(up1, end3)
        end5 = self.conv5(torch.cat([a1, up1], axis=1),t)
        up2 = self.transp_conv2(end5)

        a2 = self.att2(up2, end2)
        end6 = self.conv6(torch.cat([a2, up2], axis=1),t)
        up3 = self.transp_conv3(end6)

        a3 = self.att3(up3, end1)
        end7 = self.conv7(torch.cat([a3, up3], axis=1),t)

        return self.out(end7)

class AttUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, base=64, dropout=0.2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transp_conv1 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.transp_conv2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.transp_conv3 = nn.ConvTranspose2d(base*2, base*1, kernel_size=2, stride=2)

        self.in_x = self.get_in_proj(in_channels, out_channels=base//4)
        
        self.conv1 = nn.Sequential(
            self.get_conv_block(base//4, base, dropout),
            self.get_conv_block(base, base, dropout)
        )
        self.conv2 = nn.Sequential(
            self.get_conv_block(base, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout)
        )
        self.conv3 = nn.Sequential(
            self.get_conv_block(base*2, base*4, dropout),
            self.get_conv_block(base*4, base*4, dropout)
        )
        self.conv4 = nn.Sequential(
            self.get_conv_block(base*4, base*8, dropout),
            self.get_conv_block(base*8, base*8, dropout)
        )
        self.conv5 = nn.Sequential(
            self.get_conv_block(base*4+base*4, base*4, dropout),
            self.get_conv_block(base*4, base*4, dropout)
        )
        self.conv6 = nn.Sequential(
            self.get_conv_block(base*2+base*2, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout)
        )
        self.conv7 = nn.Sequential(
            self.get_conv_block(base+base, base, dropout),
            self.get_conv_block(base, base, dropout)
        )

        self.att1 = AttentionGate(base*4, base*4, base*4)
        self.att2 = AttentionGate(base*2, base*2, base*2)
        self.att3 = AttentionGate(base, base, base)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.Sigmoid()
        )

    def get_conv_block(self, in_channels, out_channels, dropout_val):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val, inplace=True),
            nn.SiLU(inplace=True)
        )
    
    def get_in_proj(self, in_channels, out_channels=16):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.in_x(x)

        end1 = self.conv1(x)
        down1 = self.pool(end1)

        end2 = self.conv2(down1)
        down2 = self.pool(end2)

        end3 = self.conv3(down2)
        down3 = self.pool(end3)

        end4 = self.conv4(down3)
        up1 = self.transp_conv1(end4)

        a1 = self.att1(up1, end3)
        end5 = self.conv5(torch.cat([a1, up1], axis=1))
        up2 = self.transp_conv2(end5)

        a2 = self.att2(up2, end2)
        end6 = self.conv6(torch.cat([a2, up2], axis=1))
        up3 = self.transp_conv3(end6)

        a3 = self.att3(up3, end1)
        end7 = self.conv7(torch.cat([a3, up3], axis=1))

        return self.out(end7)

class UNet_t(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base=64, dropout=0.2):
        super().__init__()
        self.base = base
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transp_conv1 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.transp_conv2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.transp_conv3 = nn.ConvTranspose2d(base*2, base*1, kernel_size=2, stride=2)

        self.in_xt = self.get_in_proj(in_channels)
        self.in_t = self.get_in_proj(1)
        
        self.conv1 = nn.Sequential(
            self.get_conv_block(32, base, dropout),
            self.get_conv_block(base, base, dropout)
        )
        self.conv2 = nn.Sequential(
            self.get_conv_block(base, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout)
        )
        self.conv3 = nn.Sequential(
            self.get_conv_block(base*2, base*4, dropout),
            self.get_conv_block(base*4, base*4, dropout)
        )
        self.conv4 = nn.Sequential(
            self.get_conv_block(base*4, base*8, dropout),
            self.get_conv_block(base*8, base*8, dropout)
        )
        self.conv5 = nn.Sequential(
            self.get_conv_block(base*4+base*4, base*4, dropout),
            self.get_conv_block(base*4, base*4, dropout)
        )
        self.conv6 = nn.Sequential(
            self.get_conv_block(base*2+base*2, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout)
        )
        self.conv7 = nn.Sequential(
            self.get_conv_block(base+base, base, dropout),
            self.get_conv_block(base, base, dropout)
        )

        self.out = nn.Conv2d(in_channels=base, out_channels=out_channels, kernel_size=1, padding=0, padding_mode='reflect')

    def get_conv_block(self, in_channels, out_channels, dropout_val):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val, inplace=True),
            nn.SiLU(inplace=True)
        )
    
    def get_in_proj(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )

    def forward(self, in_xt, t):
        xt = self.in_xt(in_xt)
        t = self.in_t(t)
        x = torch.cat([xt,t], axis=1)

        end1 = self.conv1(x)
        down1 = self.pool(end1)

        end2 = self.conv2(down1)
        down2 = self.pool(end2)

        end3 = self.conv3(down2)
        down3 = self.pool(end3)

        end4 = self.conv4(down3)
        up1 = self.transp_conv1(end4)

        end5 = self.conv5(torch.cat([end3, up1], axis=1))
        up2 = self.transp_conv2(end5)

        end6 = self.conv6(torch.cat([end2, up2], axis=1))
        up3 = self.transp_conv3(end6)

        end7 = self.conv7(torch.cat([end1, up3], axis=1))

        return self.out(end7)

class ConvNetDiscriminator(nn.Module):
    def __init__(self, in_shape=4, base=64, dropout=0.2, chip_size=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            self.get_conv_block(4, base*4, dropout),
            self.get_conv_block(base*4, base*4, dropout)
        )
        self.conv2 = nn.Sequential(
            self.get_conv_block(base*4, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout)
        )
        self.conv3 = nn.Sequential(
            self.get_conv_block(base*2, base, dropout),
            self.get_conv_block(base, base, dropout)
        )

        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        
        self.reduce = nn.Sequential(
            nn.Conv2d(base, 4, kernel_size=1, padding=0),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.out = nn.Sequential(
            nn.Linear(in_features=int(4*((chip_size/16)**2)), out_features=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.reduce(x)
        x = self.out(x)
        return x

    def get_conv_block(self, in_channels, out_channels, dropout_val):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val, inplace=True),
            nn.LeakyReLU(inplace=True)
        )

if __name__=='__main__':
    model = ConvNet()
    print(model(torch.zeros([32,4,128,128])).shape)