import torch
import torch.nn as nn

class EMA(nn.Module):
    """
    Efficient Multi-scale Attention (EMA)
    """
    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0

        self.softmax = nn.Softmax(dim=-1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv3x3 = nn.Conv2d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(x)

        x1 = x1.view(b, -1)
        x2 = x2.view(b, -1)

        attn = self.softmax(x1 + x2).view(b, 1, h, w)
        return x.view(b // self.groups, -1, h, w) * attn
