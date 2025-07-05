
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(
    num_channels: int,
    num_groups: int = 32,
):
    if num_channels % num_groups == 0:
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
    elif num_channels % 2 == 0:
        return nn.GroupNorm(num_groups=int(num_channels / 2), num_channels=num_channels, eps=1e-6, affine=True)
    else:
        return nn.GroupNorm(num_groups=num_channels, num_channels=num_channels, eps=1e-6, affine=True)

@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)



class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
    
        self.norm1 = normalize(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
 
        self.norm2 = normalize(out_channels)
    
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in