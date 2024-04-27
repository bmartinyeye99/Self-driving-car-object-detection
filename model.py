import torch.nn as nn


class CustomBlock(nn.Module):
    def __init__(self, chin, chout, negative_slope=0.01):
        super().__init__()

        self.conv1 = nn.Conv2d(chin, chin, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(chin, chin, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(chin, chin, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(chin, chout, kernel_size=3, padding='same')

        self.bn = nn.BatchNorm2d(chin)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.leaky_relu(out + x)
        out = self.bn(out)
        out = self.maxpool(out)
        out = self.conv4(out)
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

        self.blocks = nn.ModuleList([
            CustomBlock(chin, channels, negative_slope),
            CustomBlock(channels, channels*2, negative_slope),
            CustomBlock(channels*2, channels*4, negative_slope),
            CustomBlock(channels*4, channels*8, negative_slope),
            CustomBlock(channels*8, channels*16, negative_slope),
            CustomBlock(channels*16, channels*32, negative_slope),
            CustomBlock(channels*32, channels*64, negative_slope),
        ])

        self.num_features = channels * 2**(len(self.blocks) - 1)

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
        for block in self.blocks:
            x = block(x)

        x = self.adaptive(x)
        f = x.reshape(x.size(0), -1)  # Flatten the feature map

        return self.regression(f).view(x.size(0), self.S, self.S, self.C + 5)
