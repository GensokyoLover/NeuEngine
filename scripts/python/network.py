import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from futils import  *
import os
import json
from utils import * 
import cv2
import torch
import torch.nn as nn
def GN(ch: int, groups: int = 16) -> nn.GroupNorm:
    g = min(groups, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class ResidualBlockGN(nn.Module):
    def __init__(self, ch: int, dilation: int = 1, groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, padding=dilation, dilation=dilation, bias=False)
        self.gn1 = GN(ch, groups)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, padding=dilation, dilation=dilation, bias=False)
        self.gn2 = GN(ch, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = F.relu(self.gn1(h), inplace=True)
        h = self.conv2(h)
        h = self.gn2(h)
        return F.relu(x + h, inplace=True)


class TextureEncoderDilated_NoBN(nn.Module):
    """
    输入:  (B, C, H, W)
    输出:  (B, D, H, W)
    不用 BN；用 dilation 扩大感受野（适合阴影/间接光这种大范围效应）。
    """
    def __init__(self, in_ch: int = 3, base_dim: int = 64, out_dim: int = 128, groups: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 3, 1, 1, bias=False),
            GN(base_dim, groups),
            nn.ReLU(inplace=True),
        )

        dils = [1, 2, 4, 8, 4, 2]
        self.body = nn.Sequential(*[ResidualBlockGN(base_dim, dilation=d, groups=groups) for d in dils])

        self.head = nn.Sequential(
            nn.Conv2d(base_dim, out_dim, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)
    
    
class ImageDecoder(nn.Module):

    """
    逐像素 decoder（MLP 的 Conv2d 等价形式）
    不使用 BN，适合阴影 / 间接光 / visibility 等低频效应
    """
    def __init__(self, in_ch, base_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, kernel_size=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(base_dim, base_dim, kernel_size=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(base_dim, base_dim, kernel_size=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(base_dim, 3, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
# class ImageDecoder(nn.Module):
#     """
#     输入:  (B, C, H, W)
#     输出:  (B, D, H, W)
#     不用 BN；用 dilation 扩大感受野（适合阴影/间接光这种大范围效应）。
#     """
#     def __init__(self, in_ch, base_dim):
#         super().__init__()
#         self.decoder = nn.Sequential(
#             nn.Linear(in_ch, base_dim),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(base_dim, base_dim),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(base_dim, base_dim),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(base_dim, base_dim),
#             nn.LeakyReLU(inplace=True),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.decoder(x)
#         return x
    
    
class PixelMultiheadAttention(nn.Module):
    """
    Treat each pixel as a batch element:
      Q: (N, 1, D)
      K: (N, Lk, D)
      V: (N, Lk, Dv)  (usually Dv=D)
    Output:
      out: (N, 1, D)
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # IMPORTANT: expects (B, L, E)
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, need_weights: bool = False):
        """
        Q: (N, 1, D)
        K: (N, 3, D)
        V: (N, 3, D)
        """
        out, attn_w = self.mha(Q, K, V, need_weights=need_weights, average_attn_weights=True)
        # out: (N, 1, D)
        if need_weights:
            # attn_w: (N, 1, 3)  (average over heads)
            return out, attn_w
        return out
    

class PixelMultiheadAttention(nn.Module):
    """
    Treat each pixel as a batch element:
      Q: (N, 1, D)
      K: (N, Lk, D)
      V: (N, Lk, Dv)  (usually Dv=D)
    Output:
      out: (N, 1, D)
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # IMPORTANT: expects (B, L, E)
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, need_weights: bool = False):
        """
        Q: (N, 1, D)
        K: (N, 3, D)
        V: (N, 3, D)
        """
        out, attn_w = self.mha(Q, K, V, need_weights=need_weights, average_attn_weights=True)
        # out: (N, 1, D)
        if need_weights:
            # attn_w: (N, 1, 3)  (average over heads)
            return out, attn_w
        return out
    

class ConeEncoder(nn.Module):
    """
    Treat each pixel as a batch element:
      Q: (N, 1, D)
      K: (N, Lk, D)
      V: (N, Lk, Dv)  (usually Dv=D)
    Output:
      out: (N, 1, D)
    """
    def __init__(self, input_dim=9, embed_dim: int = 128):
        super().__init__()
        self.ConeEncoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        """
        Q: (N, 1, D)
        K: (N, 3, D)
        V: (N, 3, D)
        """

        return self.ConeEncoder(x)
    
class ViewGeometryEncoder(nn.Module):
    """
    Treat each pixel as a batch element:
      Q: (N, 1, D)
      K: (N, Lk, D)
      V: (N, Lk, Dv)  (usually Dv=D)
    Output:
      out: (N, 1, D)
    """
    def __init__(self, input_dim=9, embed_dim: int = 128):
        super().__init__()
        self.ConeEncoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        """
        Q: (N, 1, D)
        K: (N, 3, D)
        V: (N, 3, D)
        """

        return self.ConeEncoder(x)
    

class MLP15to3Softmax(nn.Module):
    """
    Input:  (B, 15) or (..., 15)
    Output: (B, 3)  or (..., 3) with softmax normalization (sum to 1).
    """
    def __init__(
        self,
        hidden_dims=(128, 128, 64, 64),
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = 8 + 3 * 5
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)
    
class ResidualConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)  # 比 BN 稳定
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + identity
    
class MeanDownsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AvgPool2d(4, stride=4)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x
from torch.utils.checkpoint import checkpoint
class MultiScaleImageEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch,3,padding=1),
            nn.GELU(),
            nn.Conv2d(base_ch, base_ch, 3, 2,padding=1),
        )

        # stage 0
        self.stage0 = nn.Sequential(
            ResidualConvBlock(base_ch),
        )
        self.out0 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 1)
        )

        # stage 1
        self.down1 = MeanDownsample(base_ch)
        self.stage1 = nn.Sequential(
            ResidualConvBlock(base_ch),
            ResidualConvBlock(base_ch),
        )
        self.out2 = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.Conv2d(base_ch, base_ch, 1)
        )

        # stage 2
        self.down2 = MeanDownsample(base_ch)
        self.stage2 = nn.Sequential(
            ResidualConvBlock(base_ch),
            ResidualConvBlock(base_ch),
        )
        self.out4 = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.Conv2d(base_ch, base_ch, 1)
        )

        # stage 3
        self.down3 = MeanDownsample(base_ch)
        self.stage3 = ResidualConvBlock(base_ch)
        self.out8 = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.Conv2d(base_ch, base_ch, 1)
        )

    def forward(self, x):
        """
        x: (B, W, H, C)
        """
        # BWHC → BCHW
        x = x.permute(0, 3, 1, 2).contiguous()

        # ============================================================
        # 核心修改：针对高分辨率层使用 Checkpoint
        # ============================================================
        
        # 只有在训练时才需要节省显存（因为只有训练才存梯度）
        # if self.training:
        #     # 1. 重要：Checkpoint 的输入必须 requires_grad=True
        #     # 否则 Checkpoint 会失效，或者反向传播中断。
        #     # 因为 x 通常是来自 DataLoader 的纯数据，默认没有梯度。
        #     x.requires_grad_(True)
            
        #     # 2. 包裹 Stem 层
        #     # use_reentrant=False 是新版 PyTorch 推荐的写法，更加稳定
        #     x = checkpoint(self.stem, x, use_reentrant=False)
            
        #     # 3. 包裹 Stage0 (分辨率依然很高，建议包裹)
        #     x0 = checkpoint(self.stage0, x, use_reentrant=False)
        # else:
            # 验证/推理模式下，不需要 Checkpoint
        x = self.stem(x)
        x0 = self.stage0(x)
                
        # ============================================================
        
        I0 = self.out0(x0)

        # 后面的层分辨率已经降低了 (256, 128...)，显存占用相对较小
        # 通常不需要 Checkpoint，否则会过度拖慢速度
        
        # I↓2
        x = self.down1(x0)
        x = self.stage1(x)
        I2 = self.out2(x)

        # I↓4
        x = self.down2(x)
        x = self.stage2(x)
        I4 = self.out4(x)

        # I↓8
        x = self.down3(x)
        x = self.stage3(x)
        I8 = self.out8(x)

        return [I0, I2, I4, I8]
