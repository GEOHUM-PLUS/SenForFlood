# original from https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114
import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val:float=0.3, bn=True):
        super().__init__()
        self.conv_op = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        if bn:
            self.conv_op = nn.Sequential(self.conv_op, nn.BatchNorm2d(out_channels))
        self.conv_op = nn.Sequential(self.conv_op, 
                                     nn.Dropout(dropout_val),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0))
        if bn:
            self.conv_op = nn.Sequential(self.conv_op, nn.BatchNorm2d(out_channels))
        self.conv_op = nn.Sequential(self.conv_op, 
                                     nn.Dropout(dropout_val),
                                     nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val:float=0.3, bn=True):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, dropout_val=dropout_val, bn=bn)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val:float=0.3, bn=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_val=dropout_val, bn=bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        border = int((x2.shape[-1]-x1.shape[-1])/2)
        x = torch.cat([x1, x2[:,:, border:border+x1.shape[-1], border:border+x1.shape[-1]]], 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, dropout_val:float=0.3, base=6):
        super().__init__()
        self.down_conv1 = DownSample(in_channels, 2**(base+0), dropout_val=dropout_val, bn=False)
        self.down_conv2 = DownSample(2**(base+0), 2**(base+1), dropout_val=dropout_val)
        self.down_conv3 = DownSample(2**(base+1), 2**(base+2), dropout_val=dropout_val)
        self.down_conv4 = DownSample(2**(base+2), 2**(base+3), dropout_val=dropout_val)

        self.bottle_neck = DoubleConv(2**(base+3), 2**(base+4), dropout_val=dropout_val)

        self.up_conv1 = UpSample(2**(base+4), 2**(base+3), dropout_val=dropout_val)
        self.up_conv2 = UpSample(2**(base+3), 2**(base+2), dropout_val=dropout_val)
        self.up_conv3 = UpSample(2**(base+2), 2**(base+1), dropout_val=dropout_val)
        self.up_conv4 = UpSample(2**(base+1), 2**(base+0), dropout_val=dropout_val)

        self.out = nn.Conv2d(in_channels=2**(base+0), out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        down1, p1 = self.down_conv1(x)
        down2, p2 = self.down_conv2(p1)
        down3, p3 = self.down_conv3(p2)
        down4, p4 = self.down_conv4(p3)

        b = self.bottle_neck(p4)

        up1 = self.up_conv1(b,   down4)
        up2 = self.up_conv2(up1, down3)
        up3 = self.up_conv3(up2, down2)
        up4 = self.up_conv4(up3, down1)

        out = self.out(up4)
        out = torch.softmax(input=out, dim=1)
        return out

class UNet_N(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, n_downsampling:int=4, dropout_val:float=0.3):
        super().__init__()
        self.n_downsampling = n_downsampling
        self.dropout_val = dropout_val
        self.down_conv_dict = nn.ModuleDict({})
        for i in range(n_downsampling):
            if i == 0:
                self.down_conv_dict[f'{i+1}'] = DownSample(in_channels, 2**(6+i), dropout_val=self.dropout_val)
            else:
                self.down_conv_dict[f'{i+1}'] = DownSample(2**(5+i), 2**(6+i), dropout_val=self.dropout_val)

        self.bottle_neck = DoubleConv(2**(5+n_downsampling), 2**(6+n_downsampling), dropout_val=self.dropout_val)

        self.up_conv_dict = nn.ModuleDict({})
        for i in range(n_downsampling, 0, -1):
            self.up_conv_dict[f'{n_downsampling-i+1}'] = UpSample(2**(6+i), 2**(5+i), dropout_val=self.dropout_val)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        self.down_dict = nn.ParameterDict({})
        self.p_dict = nn.ParameterDict({})
        for i in range(self.n_downsampling):
            if i == 0:
                self.down_dict[f'{i+1}'], self.p_dict[f'{i+1}'] = self.down_conv_dict[f'{i+1}'](x)
            else:
                self.down_dict[f'{i+1}'], self.p_dict[f'{i+1}'] = self.down_conv_dict[f'{i+1}'](self.p_dict[f'{i}'])

        b = self.bottle_neck(self.p_dict[f'{i+1}'])

        up_dict = nn.ParameterDict({})
        for i in range(self.n_downsampling):
            if i == 0:
                up_dict[f'{i+1}'] = self.up_conv_dict[f'{i+1}'](b, self.down_dict[f'{self.n_downsampling-i}'])
            else:
                up_dict[f'{i+1}'] = self.up_conv_dict[f'{i+1}'](up_dict[f'{i}'], self.down_dict[f'{self.n_downsampling-i}'])

        out = self.out(up_dict[f'{i+1}'])
        return out