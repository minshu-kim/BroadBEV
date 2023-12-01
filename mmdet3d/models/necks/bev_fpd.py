# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn as nn
from mmdet.models import NECKS
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

__all__ = ["BEVFPDNeck"]


@NECKS.register_module()
class BEVFPDNeck(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(
        self,
        in_channels=[128, 128, 256],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        conv_cfg=dict(type="Conv2d", bias=False),
        use_conv_for_no_stride=False,
        init_cfg=None,
    ):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(BEVFPDNeck, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        self.conv = nn.Conv2d(sum(in_channels), out_channels[0], 3, padding=1, bias=False)

        if init_cfg is None:
            self.init_cfg = [
                dict(type="Kaiming", layer="ConvTranspose2d"),
                dict(type="Constant", layer="NaiveSyncBatchNorm2d", val=1.0),
            ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
            ups[0]: (N, 256, H, W)
            ups[1]: (N, 256, H, W)
        """
        assert len(x) == len(self.in_channels)

        ups = [x[0]]
        for i, inp in enumerate(x[1:]):
            inp = F.interpolate(inp, size=x[0].shape[-2:], mode='bilinear', align_corners=True)
            ups.append(inp)

        out = torch.cat(ups, dim=1)
        out = self.conv(out)

        return [out]
