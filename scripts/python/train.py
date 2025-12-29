import torch
import torch.nn as nn
import numpy
import os
import json
import pyexr
from impostor import *
import cv2
import zstandard as zstd
import pickle
from torch.utils.tensorboard import SummaryWriter
def pack_emission_data(path,output_path):
    data = None
    for i in range(100000):
        file_name = "emission_{:05d}.exr".format(i)
        if os.path.exists(path + file_name):
            img = pyexr.read(path + file_name)
            img = img.astype(numpy.float32)
            ## downsample
            #img_down = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            if i == 0:
                data = img[numpy.newaxis,...]
            else:
                data = numpy.concatenate([data,img[numpy.newaxis,...]],axis=0)
    data = data[...,:3]
    compressed = zstd.ZstdCompressor(level=10).compress(
        pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    )

    with open(output_path, "wb") as f:
        f.write(compressed)

    print(f"✅ Saved compressed file: {output_path}, size={len(compressed)/1024/1024:.2f} MB")

def load_emission_data(path="emission_data.zst"):
    with open(path, "rb") as f:
        compressed = f.read()

    decompressed = zstd.ZstdDecompressor().decompress(compressed)
    data = pickle.loads(decompressed)
    return data

def load_impostor_data(path):
    data = {}
    with open(path + "position.json","r") as file:
        position_data = json.load(file)
    data["position"] = torch.tensor(numpy.array(position_data)).float().cuda()
    data["position"] = data["position"] / data["position"].norm(dim=-1,keepdim=True)
    with open(path + "faces.json","r") as file:
        face_data = json.load(file)
    data["face"] = torch.tensor(numpy.array(face_data)).int().cuda()
    image_path = path + r"emission_16.zst"
    if os.path.exists(image_path):
        data["emission"] = torch.tensor(load_emission_data(image_path)).cuda()
    else:
        pack_emission_data(path,image_path)
        data["emission"] = torch.tensor(load_emission_data(image_path)).cuda()
    return data



class LatentDecoder(nn.Module):
    def __init__(self, latent_dim=256, base_channels=256):
        super().__init__()

        # 1. 通过全连接把 latent 换成 CNN 的起始块
        self.fc = nn.Linear(latent_dim, base_channels * 4 * 4)

        # 2. 逐层上采样 (transposed conv)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels//2, 4, stride=2, padding=1),  # 4→8
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels//2, base_channels//4, 4, stride=2, padding=1),  # 8→16
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels//4, base_channels//8, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels//8, base_channels//16, 4, stride=2, padding=1),  # 32→64
            nn.ReLU(True),

            nn.Conv2d(base_channels//16, 3, 3, padding=1),  # 输出 RGB
            nn.LeakyReLU(0.1)   # 若数据是 [0,1] 范围
        )

    def forward(self, z):
        B = z.shape[0]
        x = self.fc(z).view(B, -1, 4, 4)  # [B, C, 4, 4]
        img = self.decoder(x)             # [B, 3, 64, 64]
        return img
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


# ----------------------------------------------------------
# 解码器： latent → 16×16 image
# ----------------------------------------------------------
class LatentDecoder16(nn.Module):
    def __init__(self, latent_dim=256, base_channels=512):
        super().__init__()

        # latent → 初始 4×4 feature map
        self.fc = nn.Linear(latent_dim, base_channels * 4 * 4)

        # 4×4 → 8×8
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base_channels // 2),
        )

        # 8×8 → 16×16
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4,
                               kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base_channels // 4),
        )

        # 输出层：卷到 3 通道
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 4, base_channels // 8, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base_channels // 8),
            nn.Conv2d(base_channels // 8, 3, kernel_size=1),  # 最终 RGB
        )

    def forward(self, z):
        B = z.shape[0]

        # MLP 部分
        x = self.fc(z).view(B, -1, 4, 4)

        # CNN + 上采样
        x = self.up1(x)   # 8×8
        x = self.up2(x)   # 16×16
        x = self.final(x)

        return x  # [B, 3, 16,16]

import math
def pack_to_atlas(data, tile_h=64, tile_w=64):
    """
    data: (N, tile_h, tile_w, 3) 的 numpy 数组
    返回: 一张拼好的大图 big_img (H_big, W_big, 3)
    """
    assert data.ndim == 4 and data.shape[1] == tile_h and data.shape[2] == tile_w and data.shape[3] == 3
    N = data.shape[0]

    # 尽量接近正方形的布局
    cols = int(math.ceil(math.sqrt(N)))
    rows = int(math.ceil(N / cols))

    H_big = rows * tile_h
    W_big = cols * tile_w

    big_img = np.zeros((H_big, W_big, 3), dtype=data.dtype)

    for idx in range(N):
        r = idx // cols
        c = idx % cols
        y0, y1 = r * tile_h, (r + 1) * tile_h
        x0, x1 = c * tile_w, (c + 1) * tile_w
        big_img[y0:y1, x0:x1, :] = data[idx]

    return big_img


class LatentSpaceInterpolator(nn.Module):
    def __init__(self, source_impostor_data, target_impostor_data, precomputed):
        super().__init__()
        self.source_impostor_data = source_impostor_data
        self.target_impostor_data = target_impostor_data
        self.precomputed = precomputed
        V = source_impostor_data[0]["position"].shape[0]
        self.sphere_latent_feature = nn.Parameter(
            0.01 * torch.randn(2,V, 256),
            requires_grad=True
        )

        self.decoder = LatentDecoder16()
        self.radius = torch.Tensor([1/8,1])
    def forward(self,i,scale):
        # d=[B,3], verts=[N,3], faces=[M,3]
        info = self.precomputed[scale]

        tri_l = info["tri_lower"]
        tri_u = info["tri_upper"]
        bary_l = info["bary_lower"]
        bary_u = info["bary_upper"]
        t = info["scale_factor"]

        latent = self.sphere_latent_feature  # [2, N, 256]

        # lower
        la = latent[0][tri_l[:,0]]
        lb = latent[0][tri_l[:,1]]
        lc = latent[0][tri_l[:,2]]
        f_l = la * bary_l[:,0:1] + lb * bary_l[:,1:2] + lc * bary_l[:,2:3]

        # upper
        la = latent[1][tri_u[:,0]]
        lb = latent[1][tri_u[:,1]]
        lc = latent[1][tri_u[:,2]]
        f_u = la * bary_u[:,0:1] + lb * bary_u[:,1:2] + lc * bary_u[:,2:3]

        # radius-interpolation
        f = f_l * (1-t) + f_u * t    # [B,256]
       
        out_image = self.decoder(f).permute(0,2,3,1)
        gt_image = self.target_impostor_data[scale]["emission"]
        loss = torch.abs(out_image - gt_image).mean()
        _,w,h,_ = gt_image.shape
        #print(w,h)
        if i %300 == 0:
            if i == 0:
                gt_image_big = pack_to_atlas(gt_image.detach().cpu().numpy(),w,h)
                pyexr.write("output/gt_impostor__{}_{:05d}.exr".format(scale,i//100), gt_image_big)
                gt_source_image_big = pack_to_atlas(self.source_impostor_data[scale]["emission"].detach().cpu().numpy(),w,h)
                pyexr.write("output/gt_source_impostor__{}_{:05d}.exr".format(scale,i//100), gt_source_image_big)
            out_image_big = pack_to_atlas(out_image.detach().cpu().numpy(),w,h)
            pyexr.write("output/pred_impostor__{}_{:05d}.exr".format(scale,i//100), out_image_big)
        return loss
import time


def precompute_interpolation_info(target_data, source_verts, source_faces, radius):
    """
    target_data: list of dict，每个 scale_level 内含 position[B,3]
    source_verts: [N,3]
    source_faces: [M,3]
    radius: tensor [2] 例如 [1/8, 1]

    返回：
    info = {
        scale: {
            "tri_lower": [B,3],
            "bary_lower": [B,3],
            "tri_upper": [B,3],
            "bary_upper": [B,3],
            "scale_factor": [B,1]
        }
    }
    """
    info = {}

    for scale, t in enumerate(target_data):
        pos = t["position"]                         # [B,3]

        # 1. 找到两个球面的半径
        r = pos.norm(dim=-1, keepdim=True)          # [B,1]
        r0, r1 = radius[0], radius[1]

        # lower / upper sphere 插值权重
        t = (r - r0) / (r1 - r0)
        t = t.clamp(0,1)

        # 把方向投影到 lower sphere / upper sphere
        dir_l = pos / r * r0
        dir_u = pos / r * r1

        # 2. 计算 barycentric（改成 batch 版本）
        tri_l, bary_l = find_containing_triangle_batch(dir_l, source_verts, source_faces)
        tri_u, bary_u = find_containing_triangle_batch(dir_u, source_verts, source_faces)

        info[scale] = {
            "tri_lower": tri_l,
            "bary_lower": bary_l,
            "tri_upper": tri_u,
            "bary_upper": bary_u,
            "scale_factor": t,
        }

    return info

path = r"H:\Falcor\media\inv_rendering_scenes\light_collection_128_{}/level{}/"
source_level, target_level = 1,1
scale = [1,2,3,4,5,6,7,8]
source_data = []
target_data = []

for scale_level in scale:
    source_path = path.format(scale_level,source_level)
    target_path = path.format(scale_level,target_level)
    source_data_single = load_impostor_data(source_path)
    target_data_single = load_impostor_data(target_path)
    source_data_single["position"] = source_data_single["position"] * (9-scale_level) / 8
    target_data_single["position"] = target_data_single["position"] * (9-scale_level) / 8 
    source_data.append(source_data_single)
    target_data.append(target_data_single)

info = precompute_interpolation_info(target_data, source_data[0]["position"], source_data[0]["face"], torch.Tensor([1/8,1]).cuda())
model = LatentSpaceInterpolator(source_data, target_data,info).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_accum = 0.0
log_interval = 2400   # 每100 step 记录一次
writer = SummaryWriter(log_dir="runs/16scale0_source8_target4")
torch.cuda.synchronize()
start = time.perf_counter()

cnt = 0
for i in range(1000000):
    for scale in [0,1,2,3,4,5,6,7]:
        optimizer.zero_grad()
        loss = model(i,scale)       # forward
        
        loss.backward()      # backward
        optimizer.step()     # optimizer update
        print(i,scale,loss)
        cnt = cnt + 1
        loss_accum = loss_accum+ loss.item()
        if cnt % log_interval == 0:
            avg_loss = loss_accum / cnt
            writer.add_scalar("Loss/train_avg100", avg_loss, i)
            print(f"[Step {i}] avg_loss={avg_loss:.6f}")
            loss_accum = 0.0     # reset
            cnt = 0
torch.cuda.synchronize()
end = time.perf_counter()

elapsed = end - start

print(f"🟢 1000 次 forward+backward 总耗时：{elapsed:.4f} 秒")
print(f"🟢 每次 iteration: {elapsed/1000.0:.6f} 秒")
print(f"🟢 训练 FPS: {1000/elapsed:.2f} iters/sec")
print(f"最新 loss: {loss.item():.6f}")
