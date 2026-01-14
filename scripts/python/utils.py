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


def oct_encode(n: torch.Tensor) -> torch.Tensor:
    """
    n: (..., 3) 方向向量（不要求已归一化）
    return: (..., 2) in [0, 1]
    """
    # L1 归一化
    denom = n.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    n = n / denom

    # mask: n.z < 0
    mask = n[..., 2] < 0

    # 计算折叠项
    nx, ny = n[..., 0], n[..., 1]
    folded_x = (1.0 - ny.abs()) * nx.sign()
    folded_y = (1.0 - nx.abs()) * ny.sign()

    # 条件替换 n.xy
    n_xy = torch.stack([nx, ny], dim=-1)
    folded_xy = torch.stack([folded_x, folded_y], dim=-1)

    n_xy = torch.where(mask.unsqueeze(-1), folded_xy, n_xy)

    # [-1,1] → [0,1]
    uv = n_xy * 0.5 + 0.5
    return uv

def get_nearest_impostor_view_batch(impostor, dirs):
    """
    dirs: (B, 3)
    return: (B, 3)  view triplets
    """
    uv = oct_encode(dirs)   # (B, 2)

    res = impostor.baseCameraResolution
    ij = (uv * res).long()
    ij = torch.clamp(ij, 0, res - 1)

    # 拆成 x/y
    ix = ij[:, 0]
    iy = ij[:, 1]

    view_idx = impostor.texFaceIndex[iy, ix]  # (B,)
    return impostor.cFace[view_idx]            # (B, 3)

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


def sample_impostor_features(emissive_features, view_idx, sample_uv, sphere_radius):
    """
    根据 view_idx 和 uv 对 impostor 特征进行采样。
    
    参数:
        emissive_features: (NumViews, C, H_feat, W_feat) 
                           例如 (12, 64, 512, 512)，包含所有视角的特征图。
        view_idx: (B, W, H) 或 (B, W, H, 1)
                  每个像素对应的 View 索引 (0 ~ NumViews-1)。
                  注意：如果你的输入是 (1, W, H, 3)，请先取第一维变成整数索引。
        sample_uv: (B, W, H, 2)
                   计算出的采样坐标 (u, v)，单位是世界坐标距离。
        sphere_radius: (B, W, H, 1) 或 scalar
                   Impostor 的包围球半径，用于归一化 UV。
    
    返回:
        sampled_features: (B, W, H, C) 采样后的特征。
    """
    
    # 1. 维度检查与准备
    # 确保 view_idx 是单纯的索引 (B, W, H)
    if view_idx.dim() == 4 and view_idx.shape[-1] == 3:
        view_idx = view_idx[..., 0] # 取出索引维度
    
    # 确保 emissive_features 是 (Batch, C, Depth, Height, Width)
    # 我们把 12 个 View 当作 Depth 维度
    # 输入: (NumViews, C, Hf, Wf) -> 扩充为 -> (1, C, NumViews, Hf, Wf)
    num_views, C, Hf, Wf = emissive_features.shape
    # 这里的 Batch=1 是指特征图本身的 Batch，不是渲染画面的 Batch
    input_tensor = emissive_features.unsqueeze(0).permute(0, 2, 1, 3, 4) 
    
    # 2. UV 归一化 (Normalize UV)
    # grid_sample 需要坐标在 [-1, 1] 之间
    # 你的 sample_uv 是世界坐标下的投影距离，除以半径即可归一化
    uv_normalized = sample_uv 
    
    # 3. View Index 映射到 Depth 坐标
    # grid_sample 的 Z 轴也需要 [-1, 1]
    # 映射公式 (align_corners=True): z = -1 + (2 * idx) / (max_idx)
    z_indices = view_idx.float()
    z_coords = -1.0 + (2.0 * z_indices + 1.0) / num_views
    
    # 4. 拼接 3D 采样网格 (x, y, z) <-> (u, v, view_idx)
    # grid 需要是 (Batch, D_out, H_out, W_out, 3)
    # 我们输出是一张 2D 图，所以 D_out = 1
    
    # 此时 uv_normalized: (B, W, H, 2), z_coords: (B, W, H)
    # 拼接得到 (B, W, H, 3)
    grid_3d = torch.cat([uv_normalized, z_coords], dim=-1)
    
    # 增加两个维度以适配 grid_sample 要求的 5D 输入
    # (B, W, H, 3) -> (B, W, H, 1, 3)  <-- 注意这里的 1 是为了匹配 input 的 Depth 维度逻辑
    # 实际上，我们需要把 W, H 当作 grid_sample 的 H_out, W_out
    # 这里的维度对应有些 trick，grid_sample 期待 grid 形状为 (N, Do, Ho, Wo, 3)
    # 我们的 N=B (画面的Batch), Do=1 (只需要一层), Ho=W, Wo=H
    grid_3d = grid_3d.unsqueeze(1) # (B, 1, W, H, 3)
    
    # 5. 为了匹配 grid 的 Batch Size (B)，我们需要复制 input_tensor
    # input: (1, C, 12, Hf, Wf) -> (B, C, 12, Hf, Wf)
    batch_size = view_idx.shape[0]
    input_tensor = input_tensor.expand(batch_size, -1, -1, -1, -1)
    
    # 6. 执行 5D 采样
    # mode='bilinear': 对 UV 进行双线性插值（这是我们想要的）
    # 在 Z 轴 (View) 方向，因为我们传入的是精确的整数映射坐标，
    # 理论上只会采样到对应的那个 View，不会发生 View 之间的融合（Bleeding）。
    sampled = F.grid_sample(
        input_tensor, 
        grid_3d, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=False
    )
    
    # 输出形状: (B, C, 1, W, H)
    # 调整回: (B, W, H, C)
    sampled = sampled.squeeze(2).permute(0, 2, 3, 1)
    
    return sampled


import math

def sample_impostor_features_mipmap_radius(emissive_features_list, view_idx, sample_uv, sphere_radius):
    """
    [基于覆盖半径的 MipMap 版本]
    根据 sphere_radius 决定 LOD，混合采样多层级特征。
    
    参数:
        emissive_features_list: List[Tensor]
             多层级特征列表。Level 0 为最高清。
             [Level0(NumViews, C, H0, W0), Level1(NumViews, C, H1, W1), ...]
        view_idx: (B, W, H)
        sample_uv: (B, W, H, 2) 
             已归一化到 [-1, 1] 的采样中心点。
        sphere_radius: (B, W, H, 1) 或 (B, W, H)
             UV 空间下的采样半径 (范围建议在 0.0 ~ 1.0 之间)。
             如果 radius * H_base < 1，说明采样范围小于 1 个纹素，使用 Level 0。
             如果 radius * H_base > 1，说明采样范围覆盖多个纹素，使用更高 Level。
    
    返回:
        sampled_features: (B, W, H, C)
    """
    
    # --- 1. 基础参数与维度处理 ---
    
    # 确保 view_idx 是索引
    if view_idx.dim() == 4 and view_idx.shape[-1] == 3:
        view_idx = view_idx[..., 0]
        
    # 确保 sphere_radius 维度适配
    if sphere_radius.dim() == 3:
        sphere_radius = sphere_radius.unsqueeze(-1) # (B, W, H, 1)
        
    batch_size = view_idx.shape[0]
    
    # 获取基础分辨率 (Level 0)
    base_height = float(emissive_features_list[0].shape[-2])
    num_views = emissive_features_list[0].shape[0]
    
    # --- 2. 计算 LOD (Level of Detail) ---
    
    # 自动推断缩放因子 (Scale Factor), 例如 //4 则 scale=4, //2 则 scale=2
    if len(emissive_features_list) > 1:
        h0 = emissive_features_list[0].shape[-2]
        h1 = emissive_features_list[1].shape[-2]
        scale_factor = h0 / h1
    else:
        scale_factor = 2.0 # 默认值
        
    # 计算当前半径覆盖了多少个“基础纹素”(Level 0 texels)
    # 解释: UV空间总长为2.0。如果 sphere_radius=1.0 (覆盖全图)，则 diameter=2.0。
    # 此时 texels_covered = 2.0 * (base_height / 2.0) = base_height。
    # 公式简化为: radius * base_height
    texels_covered = sphere_radius * base_height
    
    # 计算 LOD 层级
    # 当覆盖 1 个纹素时，log(1) = 0 -> Level 0
    # 当覆盖 Scale 个纹素时，log(Scale) = 1 -> Level 1
    # 添加 1e-8 防止 log(0)
    lod_level = torch.log(texels_covered + 1e-8) / math.log(scale_factor)
    
    # 限制 LOD 范围 [0, MaxLevel]
    max_level = len(emissive_features_list) - 1.0
    lod_level = torch.clamp(lod_level, min=0.0, max=max_level)
    
    # --- 3. 准备采样网格 (Grid) ---
    
    # View Index 映射 (align_corners=False)
    z_indices = view_idx.float()
    z_coords = -1.0 + (2.0 * z_indices + 1.0) / num_views

    
    # 拼接 Grid (B, 1, W, H, 3)
    # 所有层级共用此 Grid，因为 sample_uv 已经是归一化的
    grid_3d = torch.cat([sample_uv, z_coords], dim=-1)
    grid_3d = grid_3d.unsqueeze(1) 
    
    # --- 4. 三线性混合采样 (Trilinear Blending) ---
    
    final_output = 0.0
    
    # 遍历所有层级
    for i, feature in enumerate(emissive_features_list):
        # 计算混合权重: max(0, 1 - |lod - i|)
        dist = torch.abs(lod_level - i)
        weight = torch.relu(1.0 - dist) # (B, W, H, 1)
        
        # 性能优化: 如果这一层对所有像素的贡献都极小，直接跳过
        if weight.sum() < 1e-5:
            continue
            
        # 准备 input tensor
        # (NumViews, C, Hi, Wi) -> (1, C, NumViews, Hi, Wi) -> expand Batch
        curr_input = feature.unsqueeze(0).permute(0, 2, 1, 3, 4)
        curr_input = curr_input.expand(batch_size, -1, -1, -1, -1)
        
        # 执行采样
        # align_corners=False 配合之前确定的逻辑
        sampled = F.grid_sample(
            curr_input, 
            grid_3d, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )
        
        # (B, C, 1, W, H) -> (B, W, H, C)
        sampled = sampled.squeeze(2).permute(0, 2, 3, 1)
        
        # 累加加权结果
        final_output += sampled * weight

    return final_output



def generate_mipmap_chain(tensor, downsample_factor=2):
    """
    生成特征张量的多层级 Mipmap 列表。
    
    参数:
        tensor: (B, C, H, W) 或 (B, C, W, H) 的输入张量。
                注意：PyTorch 操作总是作用于最后两个维度。
        downsample_factor: int, 下采样倍数 (例如 2 或 4)。
    
    返回:
        List[Tensor]: 包含 [Level0, Level1, Level2, ...] 的列表。
                      一直下采样直到宽或高小于 downsample_factor 为止。
    """
    # 结果列表，首先包含原始分辨率 (Level 0)
    mipmaps = [tensor]
    current_tensor = tensor
    
    while True:
        # 获取当前最后两个维度 (H, W)
        h, w = current_tensor.shape[-2:]
        
        # 计算下一层的目标尺寸
        next_h = h // downsample_factor
        next_w = w // downsample_factor
        
        # 停止条件：如果下一层尺寸小于 1，或者已经无法再进行整除下采样
        if next_h < 1 or next_w < 1:
            break
            
        # 使用 avg_pool2d 进行下采样
        # kernel_size 和 stride 都设为 downsample_factor
        # 这种方式等同于对邻域像素求平均，是标准的 Mipmap 生成方式
        current_tensor = F.avg_pool2d(
            current_tensor, 
            kernel_size=downsample_factor, 
            stride=downsample_factor
        )
        
        # 如果你更倾向于由 PyTorch 自动处理非整除的边缘情况，也可以用 interpolate mode='area'
        # current_tensor = F.interpolate(current_tensor, size=(next_h, next_w), mode='area')
        
        mipmaps.append(current_tensor)
    
    return mipmaps


def generate_mipmap_chain_minmax(tensor, downsample_factor=2, mode='max'):
    """
    生成特征张量的多层级 Mipmap 列表（基于 Min 或 Max Pooling）。
    
    参数:
        tensor: (B, C, H, W) 或 (B, C, W, H) 的输入张量。
        downsample_factor: int, 下采样倍数 (例如 2)。
        mode: str, 'max' 或 'min'。
              'max': 保留局部区域的最大值（适合自发光、高光）。
              'min': 保留局部区域的最小值（适合遮挡、SDF）。
    
    返回:
        List[Tensor]: 包含 [Level0, Level1, Level2, ...] 的列表。
    """
    assert mode in ['max', 'min'], "Mode must be 'max' or 'min'"
    
    # 结果列表，Level 0
    mipmaps = [tensor]
    current_tensor = tensor
    
    while True:
        # 获取当前尺寸
        h, w = current_tensor.shape[-2:]
        
        # 计算下一层尺寸
        next_h = h // downsample_factor
        next_w = w // downsample_factor
        
        # 停止条件
        if next_h < 1 or next_w < 1:
            break
            
        # --- 核心修改 ---
        if mode == 'max':
            # Max Pooling: 取区域内最大值
            current_tensor = F.max_pool2d(
                current_tensor, 
                kernel_size=downsample_factor, 
                stride=downsample_factor
            )
        else:
            # Min Pooling: PyTorch 没有直接的 min_pool2d
            # 技巧: min(x) = -max(-x)
            current_tensor = -F.max_pool2d(
                -current_tensor, 
                kernel_size=downsample_factor, 
                stride=downsample_factor
            )
        
        mipmaps.append(current_tensor)
    
    return mipmaps

def sample_impostor_features_mipmap_dual(emissive_features_list, view_idx, sample_uv, sphere_radius, mode='nearest_max'):
    """
    [Dual Output MipMap 通用版本]
    
    功能:
    1. 计算 per-pixel LOD。
    2. 针对 LOD 的 floor 和 ceil 层级分别采样。
    3. 支持多种单层采样策略 (Bilinear 或 Neighbor-Min/Max)。
    
    参数:
        mode (str): 采样混合模式
            - 'bilinear':    标准双线性插值 (平滑)。
            - 'nearest_max': 取 UV 周围 4 个像素的最大值 (保持高光/能量)。
            - 'nearest_min': 取 UV 周围 4 个像素的最小值 (保持遮挡/边缘)。
            
    返回:
        val_floor (Tensor): LOD 向下取整层级的采样结果 (B, W, H, C)
        val_ceil  (Tensor): LOD 向上取整层级的采样结果 (B, W, H, C)
        alpha     (Tensor): 混合权重 (0.0 ~ 1.0), 表示偏向 ceil 的程度 (B, W, H, 1)
    """
    assert mode in ['bilinear', 'nearest_max', 'nearest_min'], f"Unknown mode: {mode}"
    
    # --- 1. 维度处理与基础参数 ---

    view_idx = view_idx[..., 0]
    if sphere_radius.dim() == 3:
        sphere_radius = sphere_radius.unsqueeze(-1)
        
    batch_size, W, H = view_idx.shape
    num_views = emissive_features_list[0].shape[0]
    base_height = float(emissive_features_list[0].shape[-2])
    C = emissive_features_list[0].shape[1]
    
    # --- 2. 计算 per-pixel LOD ---
    if len(emissive_features_list) > 1:
        h0 = emissive_features_list[0].shape[-2]
        h1 = emissive_features_list[1].shape[-2]
        scale_factor = h0 / h1
    else:
        scale_factor = 2.0
        
    texels_covered = sphere_radius * base_height
    lod_level = torch.log(texels_covered + 1e-8) / math.log(scale_factor)
    max_level = len(emissive_features_list) - 1.0
    lod_level = torch.clamp(lod_level, min=0.0, max=max_level)
    
    # 计算层级索引与 Alpha
    lod_floor = torch.floor(lod_level).long().repeat(3,1,1,C)
    lod_ceil = torch.clamp(lod_floor + 1, max=int(max_level))
    alpha = lod_level - lod_floor.float() # (B, W, H, 1)
    
    # --- 3. 准备 View 坐标 ---
    z_indices = view_idx.float()
    z_coords = -1.0 + (2.0 * z_indices + 1.0) / num_views
    z_coords = z_coords.unsqueeze(-1) # (B, W, H, 1)

    # --- 4. 初始化输出 Buffer ---
    dtype = emissive_features_list[0].dtype
    device = view_idx.device
    out_floor = torch.zeros(batch_size, W, H, C, device=device, dtype=dtype)
    out_ceil  = torch.zeros(batch_size, W, H, C, device=device, dtype=dtype)
    
    # --- 5. 遍历层级 ---
    for i, feature in enumerate(emissive_features_list):
        # Mask Check
        mask_is_floor = (lod_floor == i)
        mask_is_ceil  = (lod_ceil == i)
        need_sample_mask = mask_is_floor | mask_is_ceil
        
        if not need_sample_mask.any():
            continue
            
        # 准备输入特征 (B, C, Views, H, W)
        curr_feat = feature
        curr_input = curr_feat.unsqueeze(0).permute(0, 2, 1, 3, 4).expand(batch_size, -1, -1, -1, -1)
        Hi, Wi = curr_feat.shape[-2], curr_feat.shape[-1]

        # ==========================================
        # 分支 A: 标准 Bilinear 采样
        # ==========================================
        if mode == 'bilinear':
            # 构造标准 Grid (B, 1, W, H, 3)
            # 这里的 1 代表我们在 depth 维度只采一次
            grid_uv = sample_uv # (B, W, H, 2)
            grid_z = z_coords   # (B, W, H, 1)
            
            grid_combined = torch.cat([grid_uv, grid_z], dim=-1)
            grid_combined = grid_combined.unsqueeze(1) # Add dummy depth dimension for grid_sample
            
            # 采样
            sampled_raw = F.grid_sample(
                curr_input, 
                grid_combined, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            ) # Output: (B, C, 1, W, H)
            
            sampled_val = sampled_raw.squeeze(2).permute(0, 2, 3, 1) # (B, W, H, C)

        # ==========================================
        # 分支 B: Nearest Min/Max (自定义非线性)
        # ==========================================
        else:
            # 1. 计算 4 个邻域像素坐标 (Pixel Space)
            u_px = (sample_uv[..., 0] + 1.0) * Wi / 2.0 - 0.5
            v_px = (sample_uv[..., 1] + 1.0) * Hi / 2.0 - 0.5
            
            u0 = torch.floor(u_px)
            v0 = torch.floor(v_px)
            u1 = u0 + 1
            v1 = v0 + 1
            
            # 2. 转换回归一化坐标 (Normalized Space)
            def px_to_norm(px, size):
                return (torch.clamp(px, 0, size - 1) + 0.5) * 2.0 / size - 1.0

            u0_n = px_to_norm(u0, Wi)
            u1_n = px_to_norm(u1, Wi)
            v0_n = px_to_norm(v0, Hi)
            v1_n = px_to_norm(v1, Hi)
            
            # 3. 构造 4-Stack Grid
            us = torch.stack([u0_n, u1_n, u0_n, u1_n], dim=1) # (B, 4, W, H)
            vs = torch.stack([v0_n, v0_n, v1_n, v1_n], dim=1)
            zs = z_coords.permute(0,3,1,2).expand(-1, 4, -1, -1)
            
            grid_stack = torch.stack([us, vs, zs], dim=-1) # (B, 4, W, H, 3)
            
            # 4. 采样 (必须用 nearest)
            sampled_4 = F.grid_sample(
                curr_input, grid_stack, mode='nearest', padding_mode='zeros', align_corners=False
            ) # (B, C, 4, W, H)
            
            # 5. Reduce (Min 或 Max)
            if mode == 'nearest_max':
                sampled_val, _ = torch.max(sampled_4, dim=2)
            else: # nearest_min
                sampled_val, _ = torch.min(sampled_4, dim=2)
                
            sampled_val = sampled_val.permute(0, 2, 3, 1)

        # ==========================================
        # 结果填充
        # ==========================================
        if mask_is_floor.any():
            out_floor[mask_is_floor] = sampled_val[mask_is_floor]
        if mask_is_ceil.any():
            out_ceil[mask_is_ceil] = sampled_val[mask_is_ceil]

    return out_floor, out_ceil, alpha

def sample_impostor_features_mipmap_minmax(emissive_features_list, view_idx, sample_uv, sphere_radius, pooling_mode='max'):
    """
    [Min/Max MipMap 版本]
    空间采样: 获取 UV 坐标周围的 4 个邻域像素，取 Max 或 Min。
    层级混合: Linear (层级间插值)。
    
    参数:
        pooling_mode: 'max' | 'min'
    """
    
    # --- 1. 基础参数与维度处理 ---
    
    view_idx = view_idx[..., 0]
    if sphere_radius.dim() == 3:
        sphere_radius = sphere_radius.unsqueeze(-1)
        
    batch_size, W, H = view_idx.shape
    num_views = emissive_features_list[0].shape[0]
    base_height = float(emissive_features_list[0].shape[-2])
    
    # --- 2. 计算 LOD (逻辑保持不变) ---
    if len(emissive_features_list) > 1:
        h0 = emissive_features_list[0].shape[-2]
        h1 = emissive_features_list[1].shape[-2]
        scale_factor = h0 / h1
    else:
        scale_factor = 2.0
        
    texels_covered = sphere_radius * base_height
    lod_level = torch.log(texels_covered + 1e-8) / math.log(scale_factor)
    max_level = len(emissive_features_list) - 1.0
    lod_level = torch.clamp(lod_level, min=0.0, max=max_level)
    
    # --- 3. 准备 View 坐标 (Z轴) ---
    # View 坐标对所有层级通用
    z_indices = view_idx.float()
    z_coords = -1.0 + (2.0 * z_indices + 1.0) / num_views
    z_coords = z_coords.unsqueeze(-1) # (B, W, H, 1)

    # --- 4. 混合采样 ---
    final_output = 0.0
    
    for i, feature in enumerate(emissive_features_list):
        # 4.1 计算层级权重
        dist = torch.abs(lod_level - i)
        weight = torch.relu(1.0 - dist) # (B, W, H, 1)
        
        if weight.sum() < 1e-5:
            continue
            
        # 4.2 准备当前层的特征
        # (NumViews, C, Hi, Wi)
        curr_feat = feature
        Hi, Wi = curr_feat.shape[-2], curr_feat.shape[-1]
        
        # 展开 Batch 以便 grid_sample 使用
        # (1, C, NumViews, Hi, Wi) -> (B, C, NumViews, Hi, Wi)
        curr_input = curr_feat.unsqueeze(0).permute(0, 2, 1, 3, 4)
        curr_input = curr_input.expand(batch_size, -1, -1, -1, -1)
        
        # 4.3 手动计算 4 个邻域像素的坐标
        # 原理: grid_sample(align_corners=False) 的坐标变换公式为:
        # real_x = (x_norm + 1) * W / 2 - 0.5
        
        # 将当前 UV 转换到像素坐标系 (Pixel Space)
        # sample_uv: (B, W, H, 2)
        u_px = (sample_uv[..., 0] + 1.0) * Wi / 2.0 - 0.5
        v_px = (sample_uv[..., 1] + 1.0) * Hi / 2.0 - 0.5
        
        # 找到左上角 (Top-Left) 的整数坐标
        u0 = torch.floor(u_px)
        v0 = torch.floor(v_px)
        
        # 计算 4 个邻点的整数坐标 (TL, TR, BL, BR)
        # 并限制在图像范围内 [0, size-1]
        u1 = u0 + 1
        v1 = v0 + 1
        
        u0_c = torch.clamp(u0, 0, Wi - 1)
        u1_c = torch.clamp(u1, 0, Wi - 1)
        v0_c = torch.clamp(v0, 0, Hi - 1)
        v1_c = torch.clamp(v1, 0, Hi - 1)
        
        # 将这 4 个整数坐标转换回归一化坐标 [-1, 1]
        # 公式: norm = (pixel + 0.5) * 2 / W - 1
        # 我们这里加 0.5 是为了采样像素的中心
        def px_to_norm(px, size):
            return (px + 0.5) * 2.0 / size - 1.0
            
        u0_n = px_to_norm(u0_c, Wi)
        u1_n = px_to_norm(u1_c, Wi)
        v0_n = px_to_norm(v0_c, Hi)
        v1_n = px_to_norm(v1_c, Hi)
        
        # 4.4 构造 4 个 Grid 并一次性采样
        # 组合出 4 组坐标: (u0, v0), (u1, v0), (u0, v1), (u1, v1)
        # 我们在 dim=1 (Time/Depth维度 for grid_sample output) 堆叠这4个点
        # Grid Shape 目标: (B, 4, W, H, 3) -> 这样 grid_sample 输出 (B, C, 4, W, H)
        
        # 构造 u, v 列表
        # list of (B, W, H)
        us = [u0_n, u1_n, u0_n, u1_n]
        vs = [v0_n, v0_n, v1_n, v1_n]
        
        # 堆叠
        u_stack = torch.stack(us, dim=1) # (B, 4, W, H)
        v_stack = torch.stack(vs, dim=1)
        
        # z 坐标需要复制 4 份
        z_stack = z_coords.permute(0,3,1,2).expand(-1, 4, -1, -1) # (B, 4, W, H)
        
        # 最终 Grid: (B, 4, W, H, 3)
        grid_stack = torch.stack([u_stack, v_stack, z_stack], dim=-1)
        
        # 4.5 执行采样 (Nearest Mode)
        # 因为我们已经手动计算了像素中心，mode='nearest' 会精确命中该像素
        sampled_4 = F.grid_sample(
            curr_input,
            grid_stack,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )
        # 输出: (B, C, 4, W, H)
        
        # 4.6 执行 Min 或 Max 聚合
        if pooling_mode == 'max':
            # 在 dim=2 (4个邻居) 上取最大值
            sampled_val, _ = torch.max(sampled_4, dim=2) # (B, C, W, H)
        else:
            sampled_val, _ = torch.min(sampled_4, dim=2)
            
        # 调整维度 (B, C, W, H) -> (B, W, H, C)
        sampled_val = sampled_val.permute(0, 2, 3, 1)
        
        # 4.7 累加到最终结果
        final_output += sampled_val * weight

    return final_output



import torch.nn as nn


class LightweightBlender(nn.Module):
    def __init__(self, num_sources=6, feature_channels=5,  context_dim=8, hidden_dim=64):
        """
        参数:
            num_sources (int): 输入的采样源数量 (例如 6)。
            feature_channels (int): 每个采样源的通道数。
            context_dim (int): 额外的上下文信息维度 (例如 ViewDir=3, Depth=1)，如果没有则为0。
            hidden_dim (int): 隐藏层维度，越小越快。32 或 64 通常足够。
        """
        super().__init__()
        
        # 输入总维度 = 所有特征通道之和 + 上下文维度
        # 我们需要根据特征来决定它是否重要，所以特征本身也是输入
        self.input_dim = (num_sources * feature_channels) + context_dim
        
        self.num_sources = num_sources
        
        # 使用 1x1 卷积 (等价于共享权重的 MLP)
        self.net = nn.Sequential(
            # 第一层: 压缩/特征提取
            nn.Conv2d(self.input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            
            # (可选) 第二层: 增加非线性能力，如果极度追求速度可去掉
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            
            # 输出层: 输出 num_sources 个权重值
            nn.Conv2d(hidden_dim, num_sources, kernel_size=1)
        )
        
        # 初始化：让初始权重稍微趋向于均匀分布，避免训练初期梯度消失
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, features_list, context=None):
        """
        参数:
            features_list: List[Tensor] 或 Stacked Tensor
                           如果是 List: 6个 (B, C, H, W)
                           如果是 Tensor: (B, N, C, H, W)
            context: (B, context_dim, H, W) 可选，例如视角向量或 LOD map
            
        返回:
            blended_feature: (B, C, H, W) 融合后的特征
            weights: (B, N, H, W) 可视化用的权重
        """
        # 1. 整理输入数据
        if isinstance(features_list, list):
            # Stack: (B, N, C, H, W)
            stack = torch.stack(features_list, dim=1)
        else:
            stack = features_list
            
        B, N, C, H, W = stack.shape
        
        # Flatten: 将所有特征在通道维度拼接 (B, N*C, H, W)
        # 这是为了让网络能同时看到所有候选特征，从而决定谁更好
        flat_features = stack.reshape(B, N * C, H, W)
        
        # 2. 拼接上下文 (Context)
        if context is not None:
            # 确保 context 分辨率匹配，如果不匹配需要插值
            input_tensor = torch.cat([flat_features, context], dim=1)
        else:
            input_tensor = flat_features
            
        # 3. 计算原始 Logits
        logits = self.net(input_tensor) # (B, N, H, W)
        
        # 4. 生成归一化权重 (Softmax)
        # dim=1 表示在 N 个源之间进行归一化
        weights = F.softmax(logits, dim=1) 
        
        # 5. 加权融合
        # weights: (B, N, H, W) -> (B, N, 1, H, W) 用于广播
        weights_expanded = weights.unsqueeze(2)
        
        # sum( (B, N, C, H, W) * (B, N, 1, H, W) ) -> (B, C, H, W)
        # blended_feature = torch.sum(stack * weights_expanded, dim=1)
        
        return  weights_expanded




class FeatureDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        """
        输入格式: (B, C, W, H)
        输出格式: (B, C, W, H) for Emissive, (B, C, W, H) for Occlusion
        """
        super().__init__()
        self.out_channels = in_channels * 2  # 总输出通道数
        
        # 使用 1x1 卷积代替 Linear，完美适配 (B, C, W, H)
        self.net = nn.Sequential(
            # 第一层: 特征融合
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            
            # 第二层: 输出映射
            nn.Conv2d(hidden_dim, self.out_channels, kernel_size=1, bias=True),
            
            # 最终激活: 你的要求 ReLU
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()

    def _init_weights(self):
        # 获取最后一层卷积层
        last_conv = self.net[-2]
        
        # 1. 基础初始化: 防止 Dead ReLU，给所有偏置一个小正数
        nn.init.constant_(last_conv.bias, 0.01)
        
        # 2. 针对 Occlusion 的特殊初始化
        # Occlusion 位于通道的后半部分
        # 既然它后面要接 MinPool，我们将初始值设为 1.0，防止初始全 0 导致梯度无法回传
        half_dim = self.out_channels // 2
        with torch.no_grad():
            last_conv.bias[half_dim:] = 1.0

    def forward(self, x):
        """
        参数: x: (B, C, W, H)
        返回: emissive, occlusion 均为 (B, C, W, H)
        """
        # (B, C*2, W, H)
        combined = self.net(x)
        
        # 在 dim=1 (Channel维度) 进行切分
        emissive, occlusion = torch.chunk(combined, chunks=2, dim=1)
        
        return emissive, occlusion

def accumulate_feature_group_alpha(
    feature_list,  # list of [B, C, W, H]
    alpha_list
):
    _, G,_, _ = alpha_list[0].shape
    B, C, W, H = feature_list[0].shape
    
    assert C % G == 0
    Cg = C // G

    # f0 reshape → [B, G, Cg, W, H]
    out = feature_list[0].view(B, G, Cg, W, H)
    if len(alpha_list) == 1:
        return out.view(B, C, W, H)
    # alpha1 → [B, G, 1, W, H]（broadcast 到 channel）
    running_alpha = alpha_list[1].unsqueeze(2)

    for i in range(1, len(feature_list)):
        if i > 1:
            running_alpha = torch.minimum(
                running_alpha,
                alpha_list[i].unsqueeze(2)
            )

        fi = feature_list[i].view(B, G, Cg, W, H)
        alpha_stable = torch.where(running_alpha==torch.inf,0,running_alpha)
        out = out + fi * alpha_stable

    return out.view(B, C, W, H)
   

if __name__ == "__main__":
    img0 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    img1 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    img2 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    img3 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    img4 = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)

    last = show_5_images_click_single_red_block(img0, img1, img2, img3, img4, block=8)
    print("Last click:", last)


