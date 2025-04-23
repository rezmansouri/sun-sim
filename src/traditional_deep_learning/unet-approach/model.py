import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, output_ch, channels, dropout=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = output_ch
        self.dropout = dropout

        self.drop = nn.Dropout2d()

        self.inc = DoubleConv(n_channels, channels[0])
        self.outc = DoubleConv(channels[0], output_ch)
        downs, ups = [], []
        for i in range(len(channels) - 1):
            downs.append(Down(channels[i], channels[i + 1]))
            ups.append(Up(channels[-i - 1], channels[-i - 2]))
        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)

    def forward(self, x):

        skips = []

        x = self.inc(x)
        if self.dropout:
            x = self.drop(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            if self.dropout:
                x = self.drop(x)
            skips.append(x)

        skips.reverse()

        skips = skips[1:]

        for i, up in enumerate(self.ups):
            x = up(x, skips[i])
            if self.dropout:
                x = self.drop(x)

        x = self.outc(x)

        return F.sigmoid(x)
