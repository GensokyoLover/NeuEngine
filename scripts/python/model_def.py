# model_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------
# Channel Attention
# --------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Linear(c, c // r)
        self.fc2 = nn.Linear(c // r, c)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.mean(dim=[2,3])             # (B, C)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)

class RSMEncoder(nn.Module):
    """
    输入: rsm [B,7,384,64]
    输出: tokens [B,6,64]
    """
    def __init__(self, in_channels=7, embed_dim=64):
        super().__init__()
        self.pre_fc = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            nn.Conv2d(128, 256, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 512, kernel_size=4, stride=4),
            nn.ReLU(),
        )

        self.fc = nn.Linear(512, embed_dim)

    def forward(self, rsm):
        B = rsm.shape[0]

        x = rsm.permute(0, 2, 3, 1)     # [B,H,W,7]
        x = self.pre_fc(x)              # [B,H,W,64]
        x = x.permute(0, 3, 1, 2)       # [B,64,H,W]

        x = self.conv(x)                # [B,512,6,1]
        x = x.squeeze(-1).permute(0, 2, 1)  # [B,6,512]

        return self.fc(x)               # [B,6,64]

# --------------------------------------------------------
# Per-pixel Cross Attention
# --------------------------------------------------------
class PixelAttention(nn.Module):

    
    def __init__(self, dim=64, num_heads=8):
        super().__init__()

        # PyTorch 原生多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, surf_emb, rsm_emb):
        """
        surf_emb: [B, 64, H, W]     (Query)
        rsm_emb:  [B, 6,64]   (Key, Value)
        """
        B, C, H, W = surf_emb.shape


        # → flatten 为 sequence
        q = surf_emb.permute(0, 2, 3, 1).reshape(B*H*W,1, C)    
        k = (
            rsm_emb.unsqueeze(1)
            .expand(B, H*W, 6, C)
            .contiguous()  # 保证 reshape 成功
            .reshape(B*H*W, 6, C)
        )
   
        v = k.clone()

        # 多头注意力（batch_first=True）
        out, _ = self.attn(q, k, v)                             # [B, HW, 64]

        # reshape 回 feature map
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)       # [B, 64, H, W]
        return out


# --------------------------------------------------------
# The Full Model
# --------------------------------------------------------
class IndirectLightNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.surf_proj = nn.Conv2d(10, 64, 1)
        self.rsm_encoder = RSMEncoder(7, 64)
        self.att = PixelAttention(64, 8)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1)
        )

    def forward(self, rsm, surf):
        """
        rsm:  [B, 7, 384, 64]
        surf: [B,10, 512,512]
        """
        rsm_emb = self.rsm_encoder(rsm)      # B,64,384,64
        surf_emb = self.surf_proj(surf)   # B,64,512,512

        print(surf_emb.shape)
        print(rsm_emb.shape)
        
        fused = self.att(surf_emb, rsm_emb)

        out = self.out_conv(fused)
        return out  # B,64,512,512


def build_model():
    model = IndirectLightNet().eval()
    return model
