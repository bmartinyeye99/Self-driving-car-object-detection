import torch
import torch.nn as nn


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // 2, eps=1e-3, momentum=0.03),
            nn.SiLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

        self.c_out = nn.Sequential(
            nn.Conv2d(in_channels // 2 * 4, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        out = torch.cat([x, pool1, pool2, pool3], dim=1)

        return self.c_out(out)


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBL, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
            nn.SiLU()
        )

    def forward(self, x):
        return self.layers(x)


class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.01):
        super().__init__()

        self.conv_same = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding='same')

        self.conv_double = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding='same')

        self.conv_extraction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1)
        )

        self.bn = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv_same(x)
        out = self.conv_extraction(out)

        out = self.conv_same(out)
        out = self.conv_extraction(out)

        out = self.conv_same(out)

        out = self.leaky_relu(out + x)
        out = self.bn(out)
        out = self.maxpool(out)

        out = self.conv_double(out)
        return out


class C3(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super(C3, self).__init__()
        c_ = int(width_multiple*in_channels)

        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = CBL(
            in_channels,  c_, kernel_size=1, stride=1, padding=0)

        self.seq = nn.Sequential(
            CBL(c_, c_, 1, 1, 0),
            CBL(c_, c_, 3, 1, 1),

            CBL(c_, c_, 1, 1, 0),
            CBL(c_, c_, 3, 1, 1),
        )
        self.c_out = CBL(c_ * 2, out_channels,
                         kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.S = cfg.S
        self.C = cfg.C

        chin = cfg.chin
        channels = cfg.channels
        num_hidden = cfg.num_hidden
        dropout_rate = cfg.dropout_rate
        negative_slope = cfg.negative_slope

        self.layers = nn.ModuleList([
            CustomBlock(chin, channels, negative_slope),
            CustomBlock(channels, channels*2, negative_slope),
            CustomBlock(channels*2, channels*4, negative_slope),
            CustomBlock(channels*4, channels*8, negative_slope),
            CustomBlock(channels*8, channels*16, negative_slope),
            CustomBlock(channels*16, channels*32, negative_slope),
            SPPF(channels*32, channels*32),

            CBL(channels*32, channels*16, kernel_size=1, stride=1, padding=0),
            C3(channels*16, channels*8, width_multiple=0.25),
            CBL(channels*8, channels*4, kernel_size=1, stride=1, padding=0),
            C3(channels*4, channels*4, width_multiple=0.5),

            CustomBlock(channels*4, channels*8, negative_slope),
            CustomBlock(channels*8, channels*16, negative_slope),
        ])

        self.num_features = channels * 16

        self.adaptive = nn.AdaptiveAvgPool2d((1, 1))

        self.regression = nn.Sequential(
            nn.Linear(self.num_features, num_hidden),

            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden, num_hidden // 2),

            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout_rate),

            # 4 coordinates for each bounding box
            # 1 box per cell
            # S * S cells
            nn.Linear(num_hidden // 2, self.S * self.S * (self.C + 5)),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.adaptive(x)
        f = x.reshape(x.size(0), -1)  # Flatten the feature map

        out = self.regression(f)
        out = out.view(x.size(0), self.S, self.S, self.C + 5)

        return out
