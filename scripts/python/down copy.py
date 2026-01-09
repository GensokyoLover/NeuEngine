import torch
import torch.nn.functional as F
import os
import pyexr

# =====================================================
# 配置区
# =====================================================
ROOT_DIR = r"H:\Falcor\datasets\dragon\level1"

PREFIXES = [
    "depth",
    "normal",
    "albedo",
]

EXT = ".exr"

FW = 1   # W 方向下采样倍数
FH = 1   # H 方向下采样倍数

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# =====================================================
# 下采样函数
# =====================================================
def downsample_depth_minmax(x, fw, fh):
    """
    depth 专用：
    ch0 -> min
    ch1 -> max
    ch2,3 -> mean

    x: (B, W, H, 4)
    return: (B, W/fw, H/fh, 4)
    """
    # (B,W,H,4) -> (B,4,H,W)
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

    return y.permute(0, 3, 2, 1).contiguous()


def downsample_mean(x, fw, fh):
    """
    普通 mean pooling
    x: (B, W, H, 4)
    """
    x = x.permute(0, 3, 2, 1).contiguous()
    y = F.avg_pool2d(x, kernel_size=(fh, fw), stride=(fh, fw))
    return y.permute(0, 3, 2, 1).contiguous()


# =====================================================
# 输出路径
# =====================================================
def make_output_path(path):
    dir_name, base = os.path.split(path)
    name, ext = os.path.splitext(base)
    return os.path.join(dir_name, f"2{name}{ext}")


# =====================================================
# 处理单个 EXR
# =====================================================
def process_one_exr(path):
    basename = os.path.basename(path)
    print(f"[INFO] Processing {basename}")

    # ---------- read ----------
    data = pyexr.read(path)          # (W,H,C)
    W, H, C = data.shape

    # ---------- 扩展到 W,H,4 ----------
    data4 = torch.zeros((W, H, 4), dtype=DTYPE, device=DEVICE)
    data4[..., :min(C, 4)] = torch.from_numpy(data[..., :4]).to(DEVICE, DTYPE)

    x = data4.unsqueeze(0)  # (1,W,H,4)
    x = torch.nan_to_num(x, nan=1e10, posinf=1e10)

    # ---------- downsample ----------
    if basename.startswith("depth"):
        y = downsample_depth_minmax(x, FW, FH)
    else:
        y = downsample_mean(x, FW, FH)

    # ---------- write ----------
    out = y[0].detach().cpu().numpy()
    out_path = make_output_path(path)
    pyexr.write(out_path, out)


# =====================================================
# 主流程
# =====================================================
def main():
    paths = []

    for root, _, files in os.walk(ROOT_DIR):
        for f in files:
            if (
                f.lower().endswith(EXT)
                and any(f.startswith(p) for p in PREFIXES)
            ):
                paths.append(os.path.join(root, f))

    print(f"[INFO] Found {len(paths)} EXR files")

    for p in paths:
        process_one_exr(p)

    print("[INFO] Done")


if __name__ == "__main__":
    main()
