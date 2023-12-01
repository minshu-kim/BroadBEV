# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models import BACKBONES

__all__ = ["BEVFPD"]


@BACKBONES.register_module()
class BEVFPD(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(
        self,
        in_channels=128,
        out_channels=[128, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
        init_cfg=None,
        pretrained=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        
        self.enc = nn.Conv2d(in_filters[0], in_filters[0], 7, padding=3, bias=False)
        
        blocks = []
        res_blocks1 = []
        res_blocks2 = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1,
                ),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            block = nn.Sequential(*block)
            blocks.append(block)
            
            res_block1 = [
                build_conv_layer(
                    conv_cfg, out_channels[i], out_channels[i], 3, padding=1
                ),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    conv_cfg, out_channels[i], out_channels[i], 3, padding=1
                ),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            res_block1 = nn.Sequential(*res_block1)
            res_blocks1.append(res_block1)

            res_block2 = [
                build_conv_layer(
                    conv_cfg, out_channels[i], out_channels[i], 3, padding=1
                ),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    conv_cfg, out_channels[i], out_channels[i], 3, padding=1
                ),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            res_block2 = nn.Sequential(*res_block2)
            res_blocks2.append(res_block2)

        self.blocks = nn.ModuleList(blocks)
        self.res_blocks1 = nn.ModuleList(res_blocks1)
        self.res_blocks2 = nn.ModuleList(res_blocks2)

        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be setting at the same time"
        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is a deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        else:
            self.init_cfg = dict(type="Kaiming", layer="Conv2d")

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
            outs[0]: (N, 128, H, W)
            outs[0]: (N, 256, H//2, W//2)
        """
        x = self.enc(x)

        outs = [x]
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            x = F.relu(self.res_blocks1[i](x) + x, inplace=True)
            x = F.relu(self.res_blocks2[i](x) + x, inplace=True)

            outs.append(x)

        return tuple(outs)
