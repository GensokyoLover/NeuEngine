import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
##########################################
#  1. 禁用 cuDNN，使用 torch.compile
##########################################

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False   # compile 会更快

device = "cuda"

##########################################
# 2. RSM Encoder
##########################################

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


##########################################
# 3. Per-Pixel Cross Attention
##########################################

class PerPixelMaterialRSMAttention(nn.Module):
    def __init__(self, surf_channels=10, embed_dim=64, num_heads=4):
        super().__init__()

        self.embed_dim = embed_dim

        self.surf_proj = nn.Conv2d(surf_channels, embed_dim, kernel_size=1)

        self.rsm_encoder = RSMEncoder(in_channels=7, embed_dim=embed_dim)

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, rsm, surf):
        B, _, H, W = surf.shape

        # Surface → Q for every pixel
        surf_emb = self.surf_proj(surf)              # [B,64,H,W]
        surf_emb = surf_emb.permute(0,2,3,1)         # [B,H,W,64]
        Q = surf_emb.reshape(B, H*W, self.embed_dim) # [B,HW,64]

        # RSM → global tokens
        tokens = self.rsm_encoder(rsm)               # [B,6,64]
        K = self.k_proj(tokens)
        V = self.v_proj(tokens)

        # Cross Attention
        out, _ = self.attn(Q, K, V)                  # [B,HW,64]

        out = out.view(B, H, W, self.embed_dim).permute(0,3,1,2)
        return self.out_proj(out)                    # [B,64,H,W]


##########################################
# 4. FP16 + torch.compile
##########################################

model = PerPixelMaterialRSMAttention().to(device).half()

# ⚡ 重头戏：静态化 + 内核融合
model = torch.compile(model, mode="max-autotune")

##########################################
# 5. 准备测试输入（FP16）
##########################################

B = 3
rsm  = torch.randn(B, 7, 384, 64, device=device).half()
surf = torch.randn(B,10,512,512, device=device).half()

##########################################
# 6. Warmup + 正式测速
##########################################

# warmup
for _ in range(5):
    _ = model(rsm, surf)
torch.cuda.synchronize()

# benchmark
T = 20
t0 = time.time()
for _ in range(T):
    _ = model(rsm, surf)
torch.cuda.synchronize()
t1 = time.time()

print(f"[FP16 + torch.compile] Avg time = {(t1 - t0) * 1000 / T:.3f} ms")
