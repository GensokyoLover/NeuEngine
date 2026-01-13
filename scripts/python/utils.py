import torch
import cv2
import numpy as np
from typing import Optional, Tuple

def angle_between_tensors_atan2(a: torch.Tensor,
                                b: torch.Tensor,
                                dim: int = -1,
                                eps: float = 1e-8,
                                degrees: bool = False) -> torch.Tensor:
    """
    Stable angle computation for 3D vectors using atan2(||cross||, dot).

    a, b: (..., 3)
    return: (...) angles
    """


    dot = (a * b).sum(dim=dim)

    # cross along `dim` (assume dim is last or you can permute)
    # torch.cross only supports specifying dim in newer versions; this works generally:
    cross = torch.cross(a, b, dim=dim)
    cross_norm = cross.norm(dim=dim).clamp_min(eps)

    theta = torch.atan2(cross_norm, dot)  # radians
    if degrees:
        theta = theta * (180.0 / torch.pi)
    return theta

import os
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
import pyexr
@torch.no_grad()
def save_result_exr(output: torch.Tensor, gt: torch.Tensor | None, save_path: str):
    """
    Save output (and optionally gt) as EXR using pyexr.

    Args:
        output: (H,W,3) or (1,H,W,3) or (B,H,W,3)
        gt:     same as output or None
        save_path: must end with ".exr" (recommended)

    Behavior:
        - If gt is provided, saves a single EXR where the image is vertically concatenated:
            top = gt, bottom = output, shape = (2H, W, 3)
        - Values are saved as float32 (no clamping).
    """
    # Ensure extension
    if not save_path.lower().endswith(".exr"):
        save_path = save_path + ".exr"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def pick_first(x: torch.Tensor) -> torch.Tensor:
        # (B,H,W,3) -> (H,W,3)
        return x[0] if x.dim() == 4 else x

    out0 = pick_first(output).detach()
    if out0.dim() != 3 or out0.shape[-1] != 3:
        raise ValueError(f"output must be (H,W,3) (or batched), got {tuple(output.shape)}")

    if gt is not None:
        gt0 = pick_first(gt).detach()
        if gt0.shape != out0.shape:
            raise ValueError(f"gt shape {tuple(gt0.shape)} must match output shape {tuple(out0.shape)}")
        vis = torch.cat([gt0, out0], dim=0)  # (2H,W,3)
    else:
        vis = out0  # (H,W,3)

    # to numpy float32 on cpu
    vis_np = vis.to(dtype=torch.float32, device="cpu").contiguous().numpy()

    # pyexr expects (H,W,C)
    pyexr.write(save_path, vis_np)

import math



def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))


def _build_orthonormal_basis(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    n: (..., 3) assumed non-zero
    returns: t, b, n (each (..., 3)), orthonormal basis
    """
    n = _normalize(n)

    # Choose helper vector a not parallel to n
    # If |nz| < 0.999 use (0,0,1) else use (1,0,0)
    a1 = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype)
    a2 = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype)
    use_a1 = (n[..., 2].abs() < 0.999).unsqueeze(-1)  # (...,1)
    a = torch.where(use_a1, a1, a2)                   # (...,3) broadcast

    t = torch.cross(a, n, dim=-1)
    t = _normalize(t)
    b = torch.cross(n, t, dim=-1)
    return t, b, n

def ggx_theta_from_roughness(
    roughness: torch.Tensor,
    alpha_is_roughness_sq: bool = True,
    degrees: bool = True,
    energy_clamp: float = 99
) -> torch.Tensor:
    """
    roughness: (...,) tensor in [0,1]
    return: theta_99 (...,)  (default degrees)
    """
    r = roughness
    alpha = r * r if alpha_is_roughness_sq else r  # alpha = roughness^2 (common) or alpha=roughness
    theta = torch.atan(alpha * torch.sqrt(torch.tensor(energy_clamp, device=r.device, dtype=r.dtype)))  # radians
    if degrees:
        theta = theta * (180.0 / torch.pi)
    return theta
@torch.no_grad()
def sample_cone_directions_grid_torch(
    direction: torch.Tensor,
    half_angle_deg: torch.Tensor | float,
    H: int = 256,
    W: int = 256,
    angle_is_full: bool = False,
    flatten_hw: bool = False,
) -> torch.Tensor:
    """
    Uniformly sample H*W directions inside a cone around `direction`,
    uniform over solid angle (spherical cap). Deterministic stratified grid.

    Args:
      direction: (..., 3) tensor
      half_angle_deg: float or (...) tensor (degrees)
      H, W: grid resolution (default 256x256)
      angle_is_full: if True, interpret input as full angle, internally /2
      flatten_hw: if True -> return (..., H*W, 3) else (..., H, W, 3)

    Returns:
      dirs: unit direction vectors
    """
    if direction.shape[-1] != 3:
        raise ValueError(f"direction must have last dim 3, got {direction.shape}")

    device = direction.device
    dtype = direction.dtype

    # Flatten batch dims for simpler broadcasting
    batch_shape = direction.shape[:-1]
    B = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) > 0 else 1
    dir_flat = direction.reshape(B, 3)

    # half_angle handling
    if not torch.is_tensor(half_angle_deg):
        half_angle_deg = torch.tensor(half_angle_deg, device=device, dtype=dtype)
    else:
        half_angle_deg = half_angle_deg.to(device=device, dtype=dtype)

    if angle_is_full:
        half_angle_deg = half_angle_deg * 0.5

    # Broadcast angles to (B,)
    if half_angle_deg.ndim == 0:
        half_angle_deg_flat = half_angle_deg.expand(B)
    else:
        if half_angle_deg.shape != batch_shape:
            raise ValueError(f"half_angle_deg shape {half_angle_deg.shape} must match batch shape {batch_shape} (or be scalar).")
        half_angle_deg_flat = half_angle_deg.reshape(B)

    theta_max = half_angle_deg_flat * (math.pi / 180.0)   # (B,)
    cos_theta_max = torch.cos(theta_max).clamp(-1.0, 1.0) # (B,)

    # Deterministic stratified grid in [0,1]^2 using cell centers
    ii = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H  # (H,)
    jj = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W  # (W,)
    u = ii[:, None]  # (H,1)
    v = jj[None, :]  # (1,W)

    # Uniform over solid angle:
    # cos(theta) uniform in [cos(theta_max), 1]
    # cosθ = 1 - u*(1 - cosθmax)
    cos_t = 1.0 - u[None, :, :] * (1.0 - cos_theta_max[:, None, None])  # (B,H,W)
    sin_t = torch.sqrt((1.0 - cos_t * cos_t).clamp_min(0.0))             # (B,H,W)
    phi = (2.0 * math.pi) * v[None, :, :]                                # (B,H,W)

    x = sin_t * torch.cos(phi)
    y = sin_t * torch.sin(phi)
    z = cos_t

    # Build per-batch basis
    t, b, n = _build_orthonormal_basis(dir_flat)  # each (B,3)

    # Compose directions: x*t + y*b + z*n
    # (B,H,W,1) * (B,1,1,3) -> (B,H,W,3)
    dirs = (
        x[..., None] * t[:, None, None, :] +
        y[..., None] * b[:, None, None, :] +
        z[..., None] * n[:, None, None, :]
    )
    dirs = _normalize(dirs)

    # reshape back to batch
    dirs = dirs.reshape(*batch_shape, H, W, 3)

    if flatten_hw:
        dirs = dirs.reshape(*batch_shape, H * W, 3)
    return dirs

def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {img.shape}")
    if img.dtype == np.uint8:
        return img
    out = img.astype(np.float32)
    mx = float(np.nanmax(out)) if out.size else 0.0
    if mx <= 1.0 + 1e-6:
        out *= 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def compose_canvas(img0_512: np.ndarray, imgs4_512: list[np.ndarray]) -> np.ndarray:
    # Downsample right 4 images for display
    imgs_256 = [cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA) for im in imgs4_512]

    canvas = np.zeros((512, 1024, 3), dtype=np.uint8)
    canvas[:, 0:512, :] = img0_512

    canvas[0:256,   512:768,  :] = imgs_256[0]
    canvas[0:256,   768:1024, :] = imgs_256[1]
    canvas[256:512, 512:768,  :] = imgs_256[2]
    canvas[256:512, 768:1024, :] = imgs_256[3]

    # separators
    cv2.line(canvas, (512, 0), (512, 511), (255, 255, 255), 1)
    cv2.line(canvas, (512, 256), (1023, 256), (255, 255, 255), 1)
    cv2.line(canvas, (768, 0), (768, 511), (255, 255, 255), 1)
    return canvas


def compute_block_bbox(x: int, y: int, block: int, W: int, H: int):
    """
    以点击点附近为中心画 block x block（block=8）
    默认覆盖 [x-3 .. x+4], [y-3 .. y+4] 共8像素，并做边界裁剪。
    返回裁剪后的 (x0, x1, y0, y1) 其中 x1/y1 为 exclusive
    """
    half_low = (block // 2) - 1   # 8 -> 3
    half_high = block - half_low  # 8 -> 5 (exclusive end gives +4)

    x0 = x - half_low
    x1 = x + half_high
    y0 = y - half_low
    y1 = y + half_high

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    return x0, x1, y0, y1
import matplotlib.pyplot as plt

def visualize_dir_pointcloud_inline(reflect_dir: torch.Tensor, max_points: int = 50000):
    """
    reflect_dir: (H,W,3) torch tensor, unit vectors
    max_points:  点太多会慢，默认随机抽样到 5 万
    """
    assert reflect_dir.ndim == 3 and reflect_dir.shape[-1] == 3, f"expect (H,W,3), got {reflect_dir.shape}"

    pts = reflect_dir.detach().reshape(-1, 3).float().cpu().numpy()  # (N,3)
    N = pts.shape[0]

    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        pts = pts[idx]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 不指定颜色：用 matplotlib 默认颜色
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)

    ax.set_title("reflect_dir point cloud (unit sphere)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    # 让坐标轴等比例（球看起来不扁）
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max()
    mid = (pts.max(axis=0) + pts.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    plt.show()

def write_ply_pointcloud(path: str, points: np.ndarray, colors: np.ndarray | None = None):
    """
    points: (N,3) float
    colors: (N,3) uint8 optional
    """
    points = np.asarray(points, dtype=np.float32)
    assert points.ndim == 2 and points.shape[1] == 3

    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
        assert colors.shape == points.shape

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        if colors is None:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def sphere_covers_cone(
    cone_pos: torch.Tensor,        # (B,3) apex position P
    cone_dir: torch.Tensor,        # (B,3) forward direction (unit)
    cone_half_angle: torch.Tensor, # (B,) radians
    sphere_radius: torch.Tensor    # () or (B,)
):
    """
    Returns:
        mask: (B,) bool
        True  -> sphere fully covers the cone
        False -> cone leaks outside the sphere
    """
    B = cone_pos.shape[0]
    cone_pos = cone_pos.reshape(-1,3)
    cone_dir = cone_dir.reshape(-1,3)
    cone_half_angle = cone_half_angle.reshape(-1)
    if sphere_radius.ndim == 0:
        sphere_radius = sphere_radius.expand(B)

    # vector from apex to sphere center
    to_sphere = -cone_pos                      # (B,3)
    dist = torch.norm(to_sphere, dim=-1)       # |P|

    # --- condition 1: sphere in front ---
    t = (to_sphere * cone_dir).sum(dim=-1)     # axial projection
    valid_front = t > 0
    pyexr.write("t.exr", t.reshape(512,512,1).cpu().numpy())
    # --- condition 2: angular coverage ---
    # sin(alpha) <= r / |P|
    sin_alpha = torch.sin(cone_half_angle)

    # avoid numerical issues
    eps = 1e-6
    max_sin = sphere_radius / (dist + eps)

    angular_ok = sin_alpha <= max_sin

    return valid_front & angular_ok

def cone_radius(d, theta):
    return d * torch.tan(theta)

def cone_radius_deg(distance, theta_deg):
    """
    distance: (...,)
    theta_deg: (...,) cone 半角（角度制）
    """
    theta_rad = theta_deg * math.pi / 180.0
    return distance * torch.tan(theta_rad)

from skimage.metrics import structural_similarity as ssim

def find_most_similar_image(target_image_path, image_dir):
    # 读取目标图片（灰度）
    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise ValueError("目标图片无法读取")

    best_score = -1
    best_image = None

    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 分辨率检查（你已说明相同，但这里更安全）
        if img.shape != target.shape:
            continue

        score = ssim(target, img)

        if score > best_score:
            best_score = score
            best_image = filename

        print(filename,score)

    return best_image, best_score

import torch.nn.functional as F


def sample_by_uvi_bilinear_align_false(images, uvi,mode="bilinear"):
    """
    images: [N, H, W, 3]
    uvi:    [1, H, W, 3]  (u, v, i)
    return: [1, H, W, 3]
    """
    device = images.device
    N, _, _, C = images.shape
    _, H, W, _,_ = uvi.shape
    # NHWC -> NCHW
    images = images.permute(0, 3, 1, 2)   # [N,3,H,W]
    out_list = []
    # 拆 uvi
    for refer in range(3):
        uv = uvi[...,refer,0:2]
        idx = uvi[...,refer,2].long()
   

    # grid_sample 使用 [-1,1]，align_corners=False
        grid = uv 

        # 输出
        out = torch.zeros(1, C, H, W, device=device)

        # 逐 image index 采样（42 次是安全的）
        for i in range(N):
            mask = (idx == i)                 # [1,H,W]
            if not mask.any():
                continue

            # grid_sample
            sampled = F.grid_sample(
                images[i:i+1],                # [1,3,H,W]
                grid,                          # [1,H,W,2]
                mode=mode,
                padding_mode="zeros",
                align_corners=False
            )

            # 只保留属于当前 i 的像素
        
            out += sampled * mask.unsqueeze(1)
        out_list.append(out)
    out = torch.cat(out_list, dim=0)
    # NCHW -> NHWC
    return out.permute(0, 2, 3, 1)



def relative_cylinder_encoding_with_axis(
    cA, aA, rA, hA,
    cB, aB, rB, hB,
    eps=1e-6
):
    # center difference
    dc = cA - cB

    # axial / radial decomposition (relative to B axis)
    d_parallel = torch.sum(dc * aB, dim=-1)
    d_perp_vec = dc - d_parallel[..., None] * aB
    d_perp = torch.linalg.norm(d_perp_vec, dim=-1)

    # normalization scales
    s_h = hA + hB + eps
    s_r = rA + rB + eps

    # axis difference (orientation-invariant)
    axis_diff = 1.0 - torch.abs(torch.sum(aA * aB, dim=-1))

    return torch.stack([
        d_parallel / s_h,
        d_perp / s_r,
        torch.log((rA + eps) / (rB + eps)),
        torch.log((hA + eps) / (hB + eps)),
        axis_diff
    ], dim=-1)
def min_downsample_pool(x, b):
    # x: (B, H, W, 1) → (B, 1, H, W)
    x = x.permute(0, 3, 1, 2)
    x = -F.max_pool2d(-x, kernel_size=b, stride=b)
    # back to (B, H//b, W//b, 1)
    return x.permute(0, 2, 3, 1)

def mean_downsample_pool(x, b):
    # x: (B, H, W, 1) → (B, 1, H, W)
    x = x.permute(0, 3, 1, 2)
    
    # mean pooling
    x = F.avg_pool2d(x, kernel_size=b, stride=b)
    
    # back to (B, H//b, W//b, 1)
    return x.permute(0, 2, 3, 1)
import zstandard as zstd
import pickle
def load_zst(path):
    with open(path, "rb") as f:
        compressed_data = f.read()
    decompressed_data = zstd.decompress(compressed_data)
    sample = pickle.loads(decompressed_data)
    return sample
def save_checkpoint(
    path,
    epoch,
    global_step,
    emissive_encoder,
    geo_encoder,
    image_decoder,
    image_decoder2,
    mlp_weight
):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,

        "emissive_encoder": emissive_encoder.state_dict(),
        "geo_encoder": geo_encoder.state_dict(),
        "image_decoder": image_decoder.state_dict(),
        "image_decoder2": image_decoder2.state_dict(),
        "mlp_weight": mlp_weight.state_dict()
    }
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved to {path}")

def load_checkpoint(
    path,
    emissive_encoder,
    geo_encoder,
    image_decoder,
    image_decoder2,
    mlp_weight,
    optimizer=None,
    strict=True,
):
    ckpt = torch.load(path, map_location="cuda")

    emissive_encoder.load_state_dict(ckpt["emissive_encoder"], strict=strict)
    geo_encoder.load_state_dict(ckpt["geo_encoder"], strict=strict)
    image_decoder.load_state_dict(ckpt["image_decoder"], strict=strict)
    image_decoder2.load_state_dict(ckpt["image_decoder2"], strict=strict)
    mlp_weight.load_state_dict(ckpt["mlp_weight"], strict=strict)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)

    print(f"[Checkpoint] Loaded from {path} (epoch={epoch}, step={global_step})")
    return epoch, global_step


def trilinear_mipmap_sample(textures, ranges):
    """
    textures: list of tensors, each (B, C, H_l, W_l)
    ranges:   (B, H, W, 4)  -> u_start, u_end, v_start, v_end
    return:   (B, C, H, W)
    """
    device = ranges.device
    B, H, W, _ = ranges.shape
    base_res = textures[0].shape[-1]
    max_level = len(textures) - 1

    # ---- 1. compute footprint & LOD ----
    u0, u1, v0, v1 = ranges.unbind(-1)
    du = (u1 - u0).clamp(min=1e-8)

    footprint = du * base_res
    lod = torch.log(footprint) / torch.log(torch.tensor(4.0))
    print(lod.max())
    lod = lod.clamp(0, max_level - 1e-6)

    l0 = torch.floor(lod).long()
    l1 = (l0 + 1).clamp(max=max_level)
    w  = (lod - l0.float()).unsqueeze(1)  # (B,1,H,W)
    # pyexr.write(r"H:/Falcor/debug/l0.exr",l0[0].cpu().numpy())
    # pyexr.write(r"H:/Falcor/debug/l1.exr",l1[0].cpu().numpy())
    # pyexr.write(r"H:/Falcor/debug/w.exr",w[0].permute(1,2,0).cpu().numpy())
    # ---- 2. sampling center ----
    u = 0.5 * (u0 + u1)
    v = 0.5 * (v0 + v1)

    # grid_sample uses [-1,1]
    grid = torch.stack([
        u * 2 - 1,
        v * 2 - 1
    ], dim=-1)  # (B,H,W,2)

    # ---- 3. sample per level ----
    def sample_level(tex, grid):
        return F.grid_sample(
            tex, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )

    out0 = torch.zeros(
        (B, textures[0].shape[1], H, W),
        device=device
    )
    out1 = torch.zeros_like(out0)

    for level in range(len(textures)):
        mask0 = (l0 == level).unsqueeze(1)
        mask1 = (l1 == level).unsqueeze(1)

        if mask0.any():
            s = sample_level(textures[level], grid)
            out0 = torch.where(mask0, s, out0)

        if mask1.any():
            s = sample_level(textures[level], grid)
            out1 = torch.where(mask1, s, out1)

    # ---- 4. trilinear blend ----
    out = (1 - w) * out0 + w * out1
    return out
import random
def sample_between(v0, v1):
    """在两个向量/标量之间线性随机"""
    if isinstance(v0, (int, float)):
        return random.uniform(v0, v1)
    return [
        random.uniform(a, b)
        for a, b in zip(v0, v1)
    ]


def generate_random_material(config):
    """
    根据 config 生成一次随机结果：
    - 只有一个 key 的 roughness ∈ [0, 0.2]
    - 其他 key 的 roughness = 1.0
    """
    result = {}

    # 随机选择一个 key 作为低 roughness
    low_rough_key = random.choice(list(config.keys()))

    for key, attrs in config.items():
        out = {}

        # -------- baseColor --------
        bc0, bc1 = attrs["baseColor"]
        out["baseColor"] = sample_between(bc0, bc1)

        # -------- roughness --------
        if key == low_rough_key:
            out["roughness"] = random.uniform(0.0, 0.2)
        else:
            out["roughness"] = 1.0

        result[key] = out

    return result
def sample_between_step(v0, v1, step=0.01):
    """
    在 [v0, v1] 之间按 step 颗粒度随机采样
    支持标量 / 向量
    """
    def sample_scalar(a, b):
        ia = int(round(a / step))
        ib = int(round(b / step))
        return random.randint(min(ia, ib), max(ia, ib)) * step

    if isinstance(v0, (int, float)):
        return sample_scalar(v0, v1)

    return [
        sample_scalar(a, b)
        for a, b in zip(v0, v1)
    ]
def generate_random_material_from_config_step(config, step=0.01):
    result = {}

    for key, attrs in config.items():
        out = {}

        # roughness（0.01 粒度）
        r0, r1 = attrs["roughness"]
        out["roughness"] = sample_between_step(r0, r1, step)

        # baseColor（同样 0.01 粒度）
        c0, c1 = attrs["baseColor"]
        out["baseColor"] = sample_between_step(c0, c1, step)

        result[key] = out

    return result
if __name__ == "__main__":
    img0 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    img1 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    img2 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    img3 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    img4 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)

    last = show_5_images_click_single_red_block(img0, img1, img2, img3, img4, block=8)
    print("Last click:", last)


