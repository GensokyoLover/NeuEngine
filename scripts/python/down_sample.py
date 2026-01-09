import torch
import torch.nn.functional as F
import os
import pyexr
import numpy as np
import re
import math

# =====================================================
# 配置区
# =====================================================
ROOT_DIR = r"H:\Falcor\datasets\dragon\level4"

# 需要处理的前缀列表
PREFIXES = [
    "depth",
    "normal",
    "albedo",
]

EXT = ".exr"

# 下采样倍数 (例如由 1024 -> 256, 则填 4)
# 如果只想拼接不缩放，设为 1
FW = 8   
FH = 8   

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# =====================================================
# 工具函数
# =====================================================
def extract_id(filename):
    """从文件名中提取数字ID用于排序"""
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[-1])
    return -1

def downsample_depth_minmax(x, fw, fh):
    """
    depth 专用：
    ch0 -> min (通过 -max(-x) 实现)
    ch1 -> max
    ch2,3 -> mean
    """
    # (B, W, H, C) -> (B, C, H, W) 用于 PyTorch Pool 操作
    x = x.permute(0, 3, 2, 1).contiguous()

    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    x23 = x[:, 2:4]

    # min pooling
    y0 = -F.max_pool2d(-x0, kernel_size=(fh, fw), stride=(fh, fw))
    # max pooling
    y1 = F.max_pool2d(x1, kernel_size=(fh, fw), stride=(fh, fw))
    # mean pooling
    y23 = F.avg_pool2d(x23, kernel_size=(fh, fw), stride=(fh, fw))

    y = torch.cat([y0, y1, y23], dim=1)

    # 转回 (B, W, H, C)
    return y.permute(0, 3, 2, 1).contiguous()

def downsample_mean(x, fw, fh):
    """
    普通 mean pooling (用于 Normal, Albedo)
    """
    x = x.permute(0, 3, 2, 1).contiguous()
    y = F.avg_pool2d(x, kernel_size=(fh, fw), stride=(fh, fw))
    return y.permute(0, 3, 2, 1).contiguous()

# =====================================================
# 核心处理逻辑
# =====================================================
def process_prefix_group(prefix, file_list):
    """
    处理某一个前缀的所有图片，生成一张 Atlas
    """
    if not file_list:
        return

    print(f"\n[SECTION] Processing prefix: '{prefix}' ({len(file_list)} files)")
    
    # 1. 严格排序 (保证 Depth 第1张对应 Normal 第1张)
    file_list.sort(key=lambda p: extract_id(os.path.basename(p)))

    processed_blocks = []
    
    # 获取目标尺寸信息 (通过读取第一张图)
    first_path = file_list[0]
    temp_data = pyexr.read(first_path)
    src_h, src_w = temp_data.shape[:2]
    target_h = src_h // FH
    target_w = src_w // FW

    print(f"  - Source Size: {src_w}x{src_h}")
    print(f"  - Target Tile: {target_w}x{target_h}")

    # 2. 批量处理下采样
    for idx, path in enumerate(file_list):
        basename = os.path.basename(path)
        
        # 读取
        data = pyexr.read(path) # (H, W, C)
        H, W, C = data.shape
        
        # 转 Tensor (H, W, 4)
        data4 = torch.zeros((H, W, 4), dtype=DTYPE, device=DEVICE)
        # copy data
        valid_c = min(C, 4)
        data_torch = torch.from_numpy(data[..., :valid_c]).to(DEVICE, DTYPE)
        data4[..., :valid_c] = data_torch

        x = data4.unsqueeze(0) # (1, H, W, 4) - PyEXR读出来是(H,W,C)，注意此时维度
        # 注意：用户提供的 downsample 函数期望输入是 (B, W, H, 4) 还是 (B, H, W, 4)?
        # 用户的代码里 input 是 x, permute(0,3,2,1) -> (B,4,H,W)。
        # 这意味着用户的输入假设是 (B, W, H, 4) 或者 x 是 (B, H, W, 4) 但想要 W 在 dim 1?
        # 通常 pyexr 读取是 (H, W, C)。
        # 让我们修正维度以匹配用户的函数逻辑：
        # 用户函数 permute(0, 3, 2, 1) 把 dim1(W) 放到 dim3(W)，把 dim2(H) 放到 dim2(H)。
        # 通常 PyTorch Pool 期望 (B, C, H, W)。
        # 如果输入是 (B, H, W, C)，permute(0, 3, 1, 2) 变成 (B, C, H, W)。
        # 用户的 permute(0, 3, 2, 1) 似乎是把输入当成了 (B, W, H, C)? 
        # 为了安全，这里我们显式调整为 (B, W, H, C) 传入，或者修改 downsample 函数。
        # 假设保持用户函数不变：我们需要构造 (B, W, H, 4)。
        
        x = x.permute(0, 2, 1, 3) # (1, H, W, 4) -> (1, W, H, 4)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5) # 简单清洗

        # 下采样
        with torch.no_grad():
            if prefix == "depth":
                y = downsample_depth_minmax(x, FW, FH)
            else:
                y = downsample_mean(x, FW, FH)

        # y 输出是 (B, W_new, H_new, 4)
        # 我们需要转回 (H_new, W_new, 4) 用于 numpy 拼接
        y_np = y[0].permute(1, 0, 2).detach().cpu().numpy() # (H, W, 4)
        
        processed_blocks.append(y_np)

        if (idx + 1) % 10 == 0:
            print(f"  - Processed {idx + 1}/{len(file_list)}", end='\r')

    # 3. 拼接 Atlas (Grid Layout)
    count = len(processed_blocks)
    tiles_x = math.ceil(math.sqrt(count))
    tiles_y = math.ceil(count / tiles_x)
    
    atlas_w = tiles_x * target_w
    atlas_h = tiles_y * target_h
    
    print(f"\n  - Generating Atlas: {tiles_x}x{tiles_y} grid, Final Res: {atlas_w}x{atlas_h}")

    # 初始化大图 (H, W, 4)
    atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.float32)

    for i, block in enumerate(processed_blocks):
        col = i % tiles_x
        row = i // tiles_x
        
        sx = col * target_w
        ex = sx + target_w
        sy = row * target_h
        ey = sy + target_h
        
        atlas[sy:ey, sx:ex, :] = block

    # 4. 保存
    save_name = f"{prefix}_atlas.exr"
    save_path = os.path.join(ROOT_DIR, save_name)
    pyexr.write(save_path, atlas)
    print(f"  - [SAVED] {save_path}")

    # 清理显存
    del processed_blocks
    torch.cuda.empty_cache()

# =====================================================
# 主流程
# =====================================================
def main():
    print(f"[INFO] Running on {DEVICE}")
    print(f"[INFO] Downsample Factor: {FW}x{FH}")
    
    # 1. 扫描文件并按 Prefix 分组
    file_groups = {p: [] for p in PREFIXES}

    for root, _, files in os.walk(ROOT_DIR):
        for f in files:
            lower_f = f.lower()
            if not lower_f.endswith(EXT):
                continue
            if "atlas" in lower_f: # 跳过已经生成的 atlas
                continue
            
            # 检查属于哪个 prefix
            for p in PREFIXES:
                if f.startswith(p):
                    file_groups[p].append(os.path.join(root, f))
                    break # 假设一个文件只属于一个前缀

    # 2. 对每个组进行处理
    for prefix in PREFIXES:
        group = file_groups[prefix]
        if group:
            process_prefix_group(prefix, group)
        else:
            print(f"[WARN] No files found for prefix '{prefix}'")

    print("\n[INFO] All Done.")

if __name__ == "__main__":
    main()
