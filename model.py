import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_base

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ECAAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(1, 2).unsqueeze(-1)
        return x * y.expand_as(x)

class ECABlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.2):
        super().__init__()
        self.dws_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3, groups=in_channels, bias=False)
        self.norm = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.eca = ECAAttention(out_channels)
        self.dropout = nn.Dropout2d(drop_prob)
        self.droppath = DropPath(drop_prob)
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.dws_conv(x)
        out = self.norm(out)
        out = self.conv1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.eca(out)
        out = self.dropout(out)
        out = self.droppath(out)
        return out + identity

class ConvNeXtWithECA(nn.Module):
    def __init__(self, drop_prob=0.2):
        super().__init__()
        base = convnext_base(weights="DEFAULT")
        self.features = nn.Sequential(*list(base.features.children()))
        self.eca = ECABlock(256, 256, drop_prob=drop_prob)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 3:
                x = self.eca(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

class TripletNet(nn.Module):
    def __init__(self, embeddingnet):
        super().__init__()
        self.embeddingnet = embeddingnet
    def forward(self, a, p, n):
        ea = self.embeddingnet(a)
        ep = self.embeddingnet(p)
        en = self.embeddingnet(n)
        return ea, ep, en
