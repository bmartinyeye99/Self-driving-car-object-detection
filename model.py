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


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_repeats=1):
        super().__init__()

        res_layers = []
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels // 2, channels,
                              kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1)
                )
            ]
        self.layers = nn.ModuleList(res_layers)
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            residual = x

            x = layer(x)

            x = x + residual

        return x


class CustomBlock(nn.Module):
    def __init__(self, chin, chout, negative_slope=0.01):
        super().__init__()

        self.conv_same = nn.Conv2d(chin, chin, kernel_size=3, padding='same')

        self.conv_double = nn.Conv2d(
            chin, chout, kernel_size=3, padding='same')

        self.residual = ResidualBlock(chin)

        self.bn = nn.BatchNorm2d(chin)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv_same(x)
        out = self.residual(out)

        out = self.conv_same(out)
        out = self.residual(out)

        out = self.conv_same(out)

        out = self.leaky_relu(out + x)
        out = self.bn(out)
        out = self.maxpool(out)

        out = self.conv_double(out)
        return out


class Model(nn.Module):
    def __init__(
        self,
        chin,
        channels,
        num_hidden,
        S,
        C,
        dropout_rate=0.1,
        negative_slope=0.01
    ):
        super().__init__()

        self.S = S
        self.C = C

        self.layers = nn.ModuleList([
            CustomBlock(chin, channels, negative_slope),
            CustomBlock(channels, channels*2, negative_slope),
            CustomBlock(channels*2, channels*4, negative_slope),
            CustomBlock(channels*4, channels*8, negative_slope),
            CustomBlock(channels*8, channels*16, negative_slope),
            CustomBlock(channels*16, channels*32, negative_slope),
            CustomBlock(channels*32, channels*64, negative_slope),
            SPPF(channels*64, channels*64),
        ])

        self.num_features = channels * 64

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
            nn.Linear(num_hidden // 2, S * S * (C + 5)),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.adaptive(x)
        f = x.reshape(x.size(0), -1)  # Flatten the feature map

        return self.regression(f).view(x.size(0), self.S, self.S, self.C + 5)
