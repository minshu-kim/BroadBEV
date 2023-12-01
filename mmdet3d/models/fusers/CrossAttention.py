from typing import List

import torch
from torch import nn
from .scconv import SCNET

from mmdet3d.models.builder import FUSERS

import torchvision
import torch.nn.functional as F

import torchvision
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

__all__ = ["CrossAttentionFuser"]


@FUSERS.register_module()
class CrossAttentionFuser(nn.Module):
    def __init__(self, num_heads: int, in_channels: int, out_channels: int) -> None:
        super(CrossAttentionFuser, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.scale = in_channels[1] ** -0.5

        self.cam_enc = SCNET(in_channels)

        self.cam_v = nn.Linear(in_channels[1], in_channels[1]+2, bias=False)
        self.cam_qk = nn.Linear(in_channels[1], (in_channels[1]+2)*2, bias=False)
        self.cam_proj = nn.Linear(in_channels[1]+2, in_channels[1])

        self.lidar_v = nn.Linear(in_channels[1], in_channels[1]+2, bias=False)
        self.lidar_qk = nn.Linear(in_channels[1], (in_channels[1]+2)*2, bias=False)
        self.lidar_proj = nn.Linear(in_channels[1]+2, in_channels[1])

        self.fuser = nn.Conv2d(4 * in_channels[1], in_channels[1], 3, padding=1, bias=False)

        self.i = 0

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        lidar_bev, cam_bev = inputs
        B, C, H, W = lidar_bev.shape

        cam_bev = self.cam_enc(cam_bev.float())

        cam_v = (
           self.cam_v(cam_bev.flatten(2).permute(0,2,1))
            .reshape(B, H*W, self.num_heads, (C+2) // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        lidar_v = (
            self.lidar_v(lidar_bev.flatten(2).permute(0,2,1))
            .reshape(B, H*W, self.num_heads, (C+2) // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        lidar_q, lidar_k = (
            self.lidar_qk(lidar_bev.detach().flatten(2).permute(0,2,1))
            .reshape(B, H*W, 2, self.num_heads, (C+2) // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        cam_q, cam_k = (
            self.cam_qk(cam_bev.detach().flatten(2).permute(0,2,1))
            .reshape(B, H*W, 2, self.num_heads, (C+2) // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        lidar_attn = (lidar_q @ lidar_k.transpose(-2, -1)) * self.scale
        lidar_attn = lidar_attn.softmax(dim=-1)

        cam_by_lidar_attn = (lidar_attn @ cam_v).transpose(1, 2).reshape(B, H*W, C+2)
        cam_by_lidar_attn = self.cam_proj(cam_by_lidar_attn).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        cam_bev_l = cam_bev + cam_by_lidar_attn

        lidar_by_lidar_attn = (lidar_attn @ lidar_v).transpose(1, 2).reshape(B, H*W, C+2)
        lidar_by_lidar_attn = self.lidar_proj(lidar_by_lidar_attn).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        lidar_bev_l = lidar_bev + lidar_by_lidar_attn

        cam_attn = (cam_q @ cam_k.transpose(-2, -1)) * self.scale
        cam_attn = cam_attn.softmax(dim=-1)

        cam_by_cam_attn = (cam_attn @ cam_v).transpose(1, 2).reshape(B, H*W, C+2)
        cam_by_cam_attn = self.lidar_proj(cam_by_cam_attn).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        cam_bev_c = cam_bev + cam_by_cam_attn

        lidar_by_cam_attn = (cam_attn @ lidar_v).transpose(1, 2).reshape(B, H*W, C+2)
        lidar_by_cam_attn = self.lidar_proj(lidar_by_cam_attn).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        lidar_bev_c = lidar_bev + lidar_by_cam_attn

        bev = self.fuser(torch.cat([cam_bev_c, cam_bev_l, lidar_bev_c, lidar_bev_l], dim=1))

        return bev
