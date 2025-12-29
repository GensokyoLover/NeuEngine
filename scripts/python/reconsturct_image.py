import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import *
import imageio.v3 as iio
import imageio
class FeatureGridPatchNet(nn.Module):
    """
    Learnable feature grid + CNN decoder:
    - 输入: patch 中心的像素坐标 (y, x)，大小 (B, 2)
    - 输出: 对应位置的 16x16 RGB patch，大小 (B, 3, 16, 16)
    """

    def __init__(
        self,
        grid_height=32,
        grid_width=32,
        feat_dim=64,
        patch_size=16,
    ):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.feat_dim = feat_dim
        self.patch_size = patch_size

        # 可学习的 feature grid: (1, C, Hf, Wf)
        self.feature_grid = nn.Parameter(
            torch.randn(1, feat_dim, grid_height, grid_width) * 0.01
        )

        # 用 transposed conv 把 (B, C, 1, 1) 变成 (B, 3, 16, 16)
        # 1 -> 2 -> 4 -> 8 -> 16
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 3, kernel_size=3, padding=1)  # 最后一层到 RGB
        )

    def sample_feature(self, centers_xy, H_img, W_img):
        """
        根据 patch 中心在整图中的坐标 (y,x)，从 feature grid 里 bilinear 采样出一个 feature。

        Args:
            centers_xy: (B, 2) -> (y, x), 像素坐标
            H_img, W_img: 整张图像的高宽，用来把像素坐标归一化

        Returns:
            feat: (B, C) 每个 patch 对应一个 feature 向量
        """
        device = centers_xy.device
        B = centers_xy.shape[0]

        y = centers_xy[:, 0].float()  # (B,)
        x = centers_xy[:, 1].float()  # (B,)

        # 把像素坐标归一化到 [-1, 1]，与 grid_sample 的坐标系对齐
        # 注意这里假设 feature grid 和图像共享同一个“归一化坐标系”
        # 如果想让 grid 更粗，可以直接用 y / (H_img-1) 来映射；这里就简单直接映射一下：
        y_norm = 2.0 * (y / (H_img - 1.0)) - 1.0  # (B,)
        x_norm = 2.0 * (x / (W_img - 1.0)) - 1.0  # (B,)

        # grid_sample 需要 (B, H_out, W_out, 2)，这里 H_out=W_out=1
        grid = torch.stack([x_norm, y_norm], dim=-1).view(B, 1, 1, 2)  # (B,1,1,2)

        # 把 feature_grid 扩展到 batch 维度
        feat_grid = self.feature_grid.expand(B, -1, -1, -1)  # (B, C, Hf, Wf)

        # 采样: 输出 (B, C, 1, 1)
        sampled = F.grid_sample(
            feat_grid,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        feat = sampled.view(B, self.feat_dim)  # (B, C)

        return feat

    def forward(self, centers_xy, H_img, W_img):
        """
        Args:
            centers_xy: (B, 2) patch 中心 (y, x)，像素坐标（可以是 float）
            H_img, W_img: 整张图像的高宽，用于坐标归一化

        Returns:
            patch: (B, 3, patch_size, patch_size)
        """
        B = centers_xy.shape[0]

        # 1x1 feature
        feat = self.sample_feature(centers_xy, H_img, W_img)  # (B, C)
        feat_map = feat.view(B, self.feat_dim, 1, 1)         # (B, C, 1, 1)

        # 卷成 16x16 patch
        patch = self.decoder(feat_map)  # (B, 3, 16, 16)
        return patch
    

class TransformerGridPatchNet(nn.Module):
    def __init__(
        self,
        feat_dim=64,
        patch_size=16,
        img_size=128
    ):
        super().__init__()
        grid_size = img_size // patch_size
        # transformer encoder
        self.encoder = ImageTransformerToGrid(
            in_channels=3,
            hidden_dim=feat_dim,
            patch_size=img_size // grid_size,
            img_size=img_size
        )

        self.feat_dim = feat_dim
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.img_size = img_size

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, centers_xy, img):
        """
        centers_xy: (B,2)  pixel centers
        img: (1,3,256,256)
        每次 forward 动态重新生成 feature grid
        """

        B = centers_xy.shape[0]
        H_img, W_img = self.img_size, self.img_size

        # 🔥 每个 step 重新生成 feature grid（可被反向传播）
        feat_grid = self.encoder(img)      # (1, C, 32, 32)
        feat_grid = feat_grid.expand(B, -1, -1, -1)

        # normalize coords
        y = centers_xy[:, 0] / (H_img - 1) * 2 - 1
        x = centers_xy[:, 1] / (W_img - 1) * 2 - 1
        grid = torch.stack([x, y], dim=-1).view(B, 1, 1, 2)

        # sample 1x1 feature
        sampled = F.grid_sample(feat_grid, grid, align_corners=True)

        # decode to patch
        patch = self.decoder(sampled)

        return patch

class ImageTransformerToGrid(nn.Module):
    """
    Transformer，把整张图编码成 Feature Grid (C, Hf, Wf)
    例如输出 (64, 32, 32)
    """
    def __init__(self, 
        in_channels=3, 
        hidden_dim=64,
        num_layers=4,
        num_heads=8,                 patch_size=16,
        img_size=256
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.hidden_dim = hidden_dim

        # 1. patch embedding (Conv2d最快)
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )  # 输出 (B, C, 32, 32)w

        self.pos_embed = nn.Parameter(
            torch.randn(1, hidden_dim, self.grid_h, self.grid_w) * 0.01
        )

        # 2. Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, img):
        """
        输入 img: (B, 3, 256, 256)
        输出 feature_grid: (B, C, 32, 32)
        """
        # (B, H, W) → (B, C, 32, 32)
        x = self.patch_embed(img)   # (B, C, 32, 32)
        x = x + self.pos_embed

        B, C, Hf, Wf = x.shape

        # (B, C, 32, 32) → (B, 1024, C)
        x = x.flatten(2).transpose(1, 2)

        # Transformer
        x = self.transformer(x)   # (B, 1024, C)

        # reshape 回 grid
        x = x.transpose(1, 2).reshape(B, C, Hf, Wf)  # (B, C, 32, 32)

        return x

def extract_gt_patch_from_center(img, centers_xy, patch_size=16):
    """
    从 GT 图像中裁剪以 (y,x) 为中心的 patch。

    Args:
        img: (B, 3, H, W)
        centers_xy: (B, 2) -> (y, x)
        patch_size: 16

    Return:
        gt_patches: (B, 3, patch_size, patch_size)
    """
    B, C, H, W = img.shape
    device = img.device

    centers_xy = centers_xy.to(device)
    cy = centers_xy[:, 0].round().long()  # (B,)
    cx = centers_xy[:, 1].round().long()  # (B,)

    half = patch_size // 2

    patches = []
    for b in range(B):
        y = cy[b].item()
        x = cx[b].item()

        top = y - half
        left = x - half

        # 边界处理
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if top + patch_size > H:
            top = H - patch_size
        if left + patch_size > W:
            left = W - patch_size

        patch = img[b, :, top:top + patch_size, left:left + patch_size]
        patches.append(patch)

    gt_patches = torch.stack(patches, dim=0)  # (B, 3, P, P)
    return gt_patches

from torchvision.utils import save_image

@torch.no_grad()
def reconstruct_full_image_transformer(model, all_positions, all_patches, img,
                           patch_size=16, H=256, W=256,
                           batch_size=4096, save_prefix="recon_step"):
    """
    重建完整大图（Ny*P × Nx*P），包括预测图和 GT 图。
    Transformer 每次重建都重新 forward。
    """

    device = img.device
    model.eval()

    # sliding window 格子数量
    Ny = H - patch_size + 1    # 241
    Nx = W - patch_size + 1    # 241
    N_total = Ny * Nx          # 58081

    print(f"Reconstructing full image with {N_total} patches...")

    # ---------------------------------------------
    # 1. 批量 model 预测所有 patch
    # ---------------------------------------------
    pred_patches = []

    for i in range(0, N_total, batch_size):
        centers = all_positions[i:i+batch_size]  # (B,2)
        pred = model(centers.float(), img)       # (B,3,16,16)
        pred_patches.append(pred)

    pred_patches = torch.cat(pred_patches, dim=0)    # (N_total,3,16,16)
    gt_patches   = all_patches                       # (N_total,3,16,16)

    # reshape 成 (Ny, Nx, 3, P, P)
    pred_patches = pred_patches.view(Ny, Nx, 3, patch_size, patch_size)
    gt_patches   = gt_patches.view(Ny, Nx, 3, patch_size, patch_size)

    # ---------------------------------------------
    # 2. 沿 X 方向拼接每一行
    # ---------------------------------------------
    pred_rows = [
        torch.cat(list(pred_patches[y, :, :, :, :]), dim=-1)   # (3, 16, Nx*16)
        for y in range(Ny)
    ]
    gt_rows = [
        torch.cat(list(gt_patches[y, :, :, :, :]), dim=-1)
        for y in range(Ny)
    ]

    # ---------------------------------------------
    # 3. 沿 Y 方向拼接（最终 3856×3856）
    # ---------------------------------------------
    pred_full = torch.cat(pred_rows, dim=-2)    # (3, Ny*16, Nx*16)
    gt_full   = torch.cat(gt_rows, dim=-2)

    # ---------------------------------------------
    # 4. 保存成图
    # ---------------------------------------------
    save_image(pred_full, f"{save_prefix}_pred.png")
    save_image(gt_full,   f"{save_prefix}_gt.png")

    print(f"[Saved] {save_prefix}_pred.png  and  {save_prefix}_gt.png")


def training_step(model, optimizer, batch, dataset):
    gt_patch = batch["patch"].cuda()     # (B,3,16,16)
    centers_xy = batch["pos"].cuda()     # (B,2)
    img_ids = batch["img_id"]            # (B,)

    # 根据每个 patch 的 img_id 找对应整张图
    imgs = torch.cat([dataset.images[i] for i in img_ids.tolist()], dim=0)  # (B,3,H,W)

    # Transformer + decoder
    pred = model(centers_xy.float(), imgs)

    loss = F.l1_loss(pred, gt_patch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def make_gt_mosaic(img, img_size=128, patch_size=16):
    """
    img: (1,3,H,W) or (3,H,W)
    return: (3, H*P, W*P)
    """
    if img.dim() == 4:
        img = img.squeeze(0)  # (3,H,W)

    C, H, W = img.shape
    P = patch_size

    # ---- unfold to get all patches ----
    # img_unfold: (3, H-P+1, W-P+1, P, P)
    uf = img.unfold(1, P, 1).unfold(2, P, 1)  
    # reorder to (H,P,W,P,C)
    # uf: (3, Hp, Wp, P, P)
    uf = uf.permute(1, 2, 0, 3, 4).contiguous()  # (H-P+1, W-P+1, 3, P, P)

    # reshape into mosaic
    mosaic = uf.permute(2, 0, 3, 1, 4).reshape(3, H*P, W*P)  # (3,2048,2048)

    return mosaic

@torch.no_grad()
def reconstruct_patch_grid_image(model, img, img_size=128, patch_size=16, batch=4096):
    """
    img: (1,3,H,W) GPU tensor
    return:
        mosaic: (3, H*patch_size, W*patch_size)  例如 2048×2048
    """

    model.eval()
    H = W = img_size
    P = patch_size

    # ---- 1. Generate all centers (H*W, 2) ----
    ys = torch.arange(0, H, device=img.device)
    xs = torch.arange(0, W, device=img.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).float()  # (H*W,2)

    # ---- 2. Predict patches, may take big memory → chunked ----
    all_patches = []

    for i in range(0, H*W, batch):
        p = pos[i : i + batch]
        pred = model(p, img)         # (B,3,16,16)
        all_patches.append(pred)

    patches = torch.cat(all_patches, dim=0)  # (H*W,3,P,P)

    # ---- 3. Reshape into grid: (H,W,3,P,P) ----
    patches = patches.view(H, W, 3, P, P)

    # ---- 4. Convert to mosaic ----
    # permute to (3, H, W, P, P)
    patches = patches.permute(2, 0, 1, 3, 4)

    # reshape:
    # (3, H, W, P, P) → (3, H*P, W*P)
    mosaic = patches.reshape(3, H * P, W * P)

    return mosaic  # (3, 2048, 2048)

if __name__ == "__main__":
    folder = r"H:\Falcor\media\inv_rendering_scenes\light_collection_128_1\level3/"

    dataset = EXRBatchPatchDataset(
        folder=folder,
        img_size=128,
        patch_size=16,
        num_patches=2000
    )

    loader = DataLoader(
        dataset,
        batch_size=1,          # 一次一张图（你可以设多张也可以）
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    model = TransformerGridPatchNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        count = 0

        for img, patches, pos in loader:

            # ---- Move to GPU ----
            img = img.cuda().squeeze(0)       # (1,3,256,256)
            patches = patches.cuda().squeeze(0)  # (2000,3,16,16)
            pos = pos.cuda().squeeze(0)          # (2000,2)

            # ---- Forward ----
            pred = model(pos, img)    # (2000,3,16,16)
            loss = torch.nn.functional.l1_loss(pred, patches)

            # ---- Backward ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Accumulate ----
            total_loss += loss.item()
            count += 1
        if (epoch % 1 == 0):   # 每个 epoch 解码一次
            print(f"[Epoch {epoch+1}] Reconstructing first 20 images...")

        for i in range(20):
            img, _, _ = dataset[i]
            gt = img.squeeze(0)  # (3,128,128)
            gt_np = gt.permute(1,2,0).cpu().numpy()   # (128,128,3)
            gt_path = f"results/epoch_{epoch+1}_img_{i}_gt.png"
            iio.imwrite(gt_path, (gt_np * 255).astype("uint8"))

            # --------------------
            # Save Mosaic 重建图
            # --------------------
            img_gpu = img.cuda()
            mosaic = reconstruct_patch_grid_image(
                model,
                img_gpu,
                img_size=128,
                patch_size=16
            )  # (3,2048,2048)

            out = mosaic.clamp(0,1).permute(1,2,0).cpu().numpy()
            mosaic_path = f"results/epoch_{epoch+1}_img_{i}_mosaic.png"
            iio.imwrite(mosaic_path, (out * 255).astype("uint8"))
        # -------- Epoch result --------
        avg_loss = total_loss / count
        print(f"Epoch [{epoch+1}/{num_epochs}]  Avg Loss = {avg_loss:.6f}")

        # (可选) 保存模型
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")