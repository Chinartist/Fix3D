from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import normalize,ResBlock
def replace_bn_with_gn(module, num_groups=32):
    """
    递归遍历模型，找到 BatchNorm 层并用 GroupNorm 替换。
    
    Args:
        module (nn.Module): 模型或子模块
        num_groups (int): GroupNorm 的组数，默认是 32
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d) or isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.BatchNorm3d):
            # 替换成 GroupNorm
            num_channels = child.num_features
            # 创建新的 GroupNorm
            if num_channels % num_groups == 0:
                gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            elif num_channels % 2 == 0:
                gn = nn.GroupNorm(num_groups=int(num_channels / 2), num_channels=num_channels)
            else:
                gn = nn.GroupNorm(num_groups=num_channels, num_channels=num_channels)
            # 替换
            setattr(module, name, gn)
        else:
            # 递归调用
            replace_bn_with_gn(child, num_groups)

class Conv2dAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        resblock = ResBlock(in_channels, out_channels)
        norm = normalize(out_channels)
        conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        super(Conv2dAct, self).__init__(resblock, norm, conv)
class UnetDecoderBlock(nn.Module):
    """A decoder block in the U-Net architecture that performs upsampling and feature fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bias: bool = True,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.conv1 = Conv2dAct(
            in_channels + skip_channels,
            out_channels,
            bias=bias,
        )

        self.conv2 = Conv2dAct(
            out_channels,
            out_channels,
            bias=bias,
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        target_height: int,
        target_width: int,
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_map = F.interpolate(
            feature_map,
            size=(target_height, target_width),
            mode=self.interpolation_mode,
        )
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        return feature_map


class UnetCenterBlock(nn.Sequential):
    """Center block of the Unet decoder. Applied to the last feature map of the encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        conv1 = Conv2dAct(
            in_channels,
            out_channels,

            bias=bias,
        )
        conv2 = Conv2dAct(
            out_channels,
            out_channels,

            bias=bias,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    """The decoder part of the U-Net architecture.

    Takes encoded features from different stages of the encoder and progressively upsamples them while
    combining with skip connections. This helps preserve fine-grained details in the final segmentation.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        bias: bool = True,
        add_center_block: bool = True,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()

        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if add_center_block:
            self.center = UnetCenterBlock(
                head_channels,
                head_channels,
                bias=bias,
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        self.blocks = nn.ModuleList()
        for block_in_channels, block_skip_channels, block_out_channels in zip(
            in_channels, skip_channels, out_channels
        ):
            block = UnetDecoderBlock(
                block_in_channels,
                block_skip_channels,
                block_out_channels,
                bias=bias,
                interpolation_mode=interpolation_mode,
            )
            self.blocks.append(block)
        self.head = nn.Conv2d(
            decoder_channels[-1],
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skip_connections = features[1:]

        x = self.center(head)
        hidden_states = [x]
        for i, decoder_block in enumerate(self.blocks):
            # upsample to the next spatial shape
            height, width = spatial_shapes[i+1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip_connection)
            hidden_states.append(x)
        head = self.head(x)
        hidden_states.append(head)
        return hidden_states
if __name__ == "__main__":
    # Example usage
    encoder_channels=[128+512,128+512,256+512,512+256]
    decoder_channels=[512,512,512,256]
    decoder = UnetDecoder(encoder_channels, decoder_channels)
    print(decoder)
    # Test the forward pass with random input
    features =[torch.randn(1, 3,64,64)]+ [torch.randn(1, c, 32 // (2 ** i), 32 // (2 ** i)) for i, c in enumerate(encoder_channels)]
    print([f.shape for f in features])
    output = decoder(features)
    print([o.shape for o in output])