import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        chin,
        channels,
        num_hidden,
        dropout_rate=0.1,
        negative_slope=0.01
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Conv2d(chin, channels, kernel_size=3, padding='same'),
            nn.Conv2d(channels, channels*2, kernel_size=3, padding='same'),
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
            nn.Linear(num_hidden // 2, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.adaptive(x)
        f = x.reshape(x.size(0), -1)  # Flatten the feature map

        return self.regression(f)
