# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# class ImageEncoder(nn.Module):
#     def __init__(self, trunk, neck, scalp=0, resnet=None):
#         super().__init__()
#         self.trunk = trunk
#         self.neck = neck
#         self.scalp = scalp
        
#         assert (
#             self.trunk.channel_list == self.neck.backbone_channel_list
#         )

#         # 引入 ResNet
#         self.resnet = resnet
        
#         if self.resnet is not None:
#             # ResNet 输出通道 [256, 512, 1024, 2048]
#             res_channels = self.resnet.channel_list
#             hiera_channels = list(reversed(self.trunk.channel_list))

#             self.res_proj_convs = nn.ModuleList([
#                 nn.Conv2d(rc, hc, 1) 
#                 for rc, hc in zip(res_channels, hiera_channels)
#             ])

#     def forward(self, sample: torch.Tensor):

#         # -------------------------
#         # 1. Hiera 全局特征
#         # -------------------------
#         hiera_feats = self.trunk(sample)
#         # from IPython import embed; embed()

#         # -------------------------
#         # 2. ResNet 局部特征（新增）
#         # -------------------------
#         if self.resnet is not None:
#             res_feats = self.resnet(sample)
#             fused_feats = []

#             # 对每一层特征做统一融合
#             for i, (r, h) in enumerate(zip(res_feats, hiera_feats)):

#                 # channel projection
#                 r_proj = self.res_proj_convs[i](r)

#                 # spatial 对齐
#                 if r_proj.shape[2:] != h.shape[2:]:
#                     r_proj = F.interpolate(
#                         r_proj, size=h.shape[2:], 
#                         mode="bilinear", align_corners=False
#                     )

#                 fused = r_proj + h
#                 # ---- 加 norm + relu ----
#                 # fused = self.fuse_norms[i](fused)
#                 # fused = F.relu(fused, inplace=True)

#                 # # ---- 残差连接（确保维度不变）----
#                 # # 与 h 保持相同通道数，输出依然是原来大小
#                 # fused = fused + h
                
#                 fused_feats.append(fused)

#             feats = fused_feats
#         else:
#             feats = hiera_feats

#         # -------------------------
#         # 3. FPNNeck
#         # -------------------------
#         features, pos = self.neck(feats)

#         if self.scalp > 0:
#             features, pos = features[:-self.scalp], pos[:-self.scalp]

#         return {
#             "vision_features": features[-1],
#             "vision_pos_enc": pos,
#             "backbone_fpn": features,
#         }


class ImageEncoder(nn.Module):
    def __init__(self, trunk, neck, scalp=0, resnet=None):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        )

        # 引入 ResNet
        self.resnet = resnet
        
        if self.resnet is not None:
            # ResNet 输出通道 [256, 512, 1024, 2048]
            res_channels = self.resnet.channel_list
            # Hiera 通道列表
            # 注意：这里是反转的，即从最低分辨率到最高分辨率的通道数
            hiera_channels_reversed = list(reversed(self.trunk.channel_list))

            # 用于将 ResNet 通道投影到 Hiera 对应的通道数
            self.res_proj_convs = nn.ModuleList([
                nn.Conv2d(rc, hc, 1) 
                for rc, hc in zip(res_channels, hiera_channels_reversed)
            ])
            
            # 融合后的特征具有 hiera_channels_reversed 的通道数。
            self.fuse_norms = nn.ModuleList([
                nn.BatchNorm2d(hc) 
                for hc in hiera_channels_reversed
            ])

    def forward(self, sample: torch.Tensor):

        # -------------------------
        # 1. Hiera 全局特征
        # -------------------------
        hiera_feats = self.trunk(sample) 
        # from IPython import embed; embed()

        # -------------------------
        # 2. ResNet 局部特征
        # -------------------------
        if self.resnet is not None:
            res_feats = self.resnet(sample)
            fused_feats = []
            
            for i, (r, h) in enumerate(zip(res_feats, hiera_feats)):
                if i >= len(self.res_proj_convs):
                    print(f"Warning: Index {i} out of bounds for self.res_proj_convs.")
                    fused_feats.append(h)
                    continue
                
                # channel projection: r_proj 通道数与 h 通道数一致
                r_proj = self.res_proj_convs[i](r)

                # spatial 对齐
                if r_proj.shape[2:] != h.shape[2:]:
                    r_proj = F.interpolate(
                        r_proj, size=h.shape[2:], 
                        mode="bilinear", align_corners=False
                    )

                # ----------------- 融合操作 -----------------
                # 原始融合：r_proj + h
                fused = r_proj + h
                
                fused = self.fuse_norms[i](fused)
                fused = F.relu(fused, inplace=True)

                fused = fused + h
                
                fused_feats.append(fused)

            feats = fused_feats
        else:
            feats = hiera_feats

        # -------------------------
        # 3. FPNNeck
        # -------------------------
        features, pos = self.neck(feats)

        if self.scalp > 0:
            features, pos = features[:-self.scalp], pos[:-self.scalp]

        return {
            "vision_features": features[-1],
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
