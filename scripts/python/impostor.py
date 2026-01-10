import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from futils import  *
import os
import json
from utils import * 
import cv2
import torch
def sampling_radii_from_aabb(aabb_min, aabb_max, fov_deg=60, margin=1.05):
    """
    从AABB计算两种相机下的采样半径：
    1. 正交相机 (orthographic)
    2. 透视相机 (perspective)
    """
    aabb_min = np.array(aabb_min, dtype=float)
    aabb_max = np.array(aabb_max, dtype=float)

    # 包围球参数
    center = (aabb_min + aabb_max) / 2
    radius_obj = np.linalg.norm(aabb_max - aabb_min) / 2

    # 正交相机半径
    radius_ortho = radius_obj * margin

    # 透视相机半径
    fov = np.deg2rad(fov_deg)
    radius_persp = radius_obj / np.sin(fov / 2) * margin

    return center, radius_obj, radius_ortho, radius_persp


def icosahedron_vertices_faces():
    """返回初始二十面体的12个顶点和20个三角面"""
    phi = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0],
        [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi],
        [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1],
        [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    faces = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]
    ], dtype=np.int32)
    return verts, faces


def midpoint(a, b, cache, verts):
    """返回边中点索引，防止重复创建顶点"""
    key = tuple(sorted((a, b)))
    if key in cache:
        return cache[key], verts
    mid = (verts[a] + verts[b]) / 2.0
    mid /= np.linalg.norm(mid)
    verts = np.vstack([verts, mid])
    idx = len(verts) - 1
    cache[key] = idx
    return idx, verts


def subdivide(verts, faces):
    """细分每个三角形为4个小三角形"""
    cache = {}
    new_faces = []
    for tri in faces:
        a, verts = midpoint(tri[0], tri[1], cache, verts)
        b, verts = midpoint(tri[1], tri[2], cache, verts)
        c, verts = midpoint(tri[2], tri[0], cache, verts)
        new_faces.extend([
            [tri[0], a, c],
            [tri[1], b, a],
            [tri[2], c, b],
            [a, b, c]
        ])
    return verts, np.array(new_faces, dtype=np.int32)


def geodesic_impostor_mesh(subdiv_level=2):
    """生成可控细分等级的 geodesic sphere"""
    verts, faces = icosahedron_vertices_faces()
    for _ in range(subdiv_level):
        verts, faces = subdivide(verts, faces)
    return verts, faces

def octahedral_encode(d):
    d = d / np.linalg.norm(d)
    absd = np.abs(d)
    s = absd.sum()
    x, y, z = d / s
    if z < 0:
        x_new = (1 - abs(y)) * np.sign(x)
        y_new = (1 - abs(x)) * np.sign(y)
        x, y = x_new, y_new
    return np.array([x * 0.5 + 0.5, y * 0.5 + 0.5])


def build_lookup_texture_speedup_chunk(verts, faces, resolution=2048, device="cuda", chunk_size=512):
    """
    GPU 加速构建 lookup texture，支持大分辨率分批计算，避免显存溢出。
    verts: (V, 3)
    faces: (F, 3)
    chunk_size: 每次计算的 tile 尺寸（建议 256~512）
    resolution: 贴图分辨率 (例如 2048)
    """
    # === 上传数据到 GPU ===
    verts = torch.as_tensor(verts, dtype=torch.float32, device=device)
    faces = torch.as_tensor(faces, dtype=torch.long, device=device)

    # === 计算每个面法线 ===
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / face_normals.norm(dim=1, keepdim=True)

    # === 构建 u,v 网格 ===
    grid = torch.arange(resolution, device=device, dtype=torch.float32)
    u, v = torch.meshgrid(grid, grid, indexing="xy")
    u = (u + 0.5) / resolution * 2 - 1
    v = (v + 0.5) / resolution * 2 - 1
    dir_z = 1 - torch.abs(u) - torch.abs(v)
    dir = torch.stack([u, v, dir_z], dim=-1)

    # 下半球折叠
    mask = dir[..., 2] < 0
    dir_xy = dir[..., :2]
    dir_xy_swapped = dir_xy[..., [1, 0]]
    dir_xy_new = (1 - torch.abs(dir_xy_swapped)) * torch.sign(dir_xy)
    dir[..., :2][mask] = dir_xy_new[mask]

    # 归一化
    dir = dir / dir.norm(dim=-1, keepdim=True)

    # === 分块处理 ===
    lookup = torch.empty((resolution, resolution), dtype=torch.int32, device=device)

    num_chunks = (resolution + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        for j in range(num_chunks):
            # 当前块范围
            y0, y1 = i * chunk_size, min((i + 1) * chunk_size, resolution)
            x0, x1 = j * chunk_size, min((j + 1) * chunk_size, resolution)

            dir_chunk = dir[y0:y1, x0:x1].reshape(-1, 3)  # (chunk^2, 3)
            dots = dir_chunk @ face_normals.T             # (chunk^2, F)
            lookup_chunk = torch.argmax(dots, dim=1).reshape(y1 - y0, x1 - x0)
            lookup[y0:y1, x0:x1] = lookup_chunk

            torch.cuda.empty_cache()  # 清理显存（可选）

    # === 返回 numpy 数组 ===
    return lookup.cpu().numpy()

def unity_style_up(fwd):
    f = normalize(fwd)
    up = np.array([0, 1, 0])
    if abs(np.dot(f, up)) > 0.99:
        up = np.array([0, 0, 1])
    return up
def compute_camera_basis(posW, target, up):
    # 前方向 (cameraW)
    cameraW = normalize(target - posW)

    # 右方向 (cameraU)
    cameraU = normalize(np.cross(cameraW, up))

    # 上方向 (cameraV)
    cameraV = normalize(np.cross(cameraU, cameraW))

    return cameraU, cameraV, cameraW

def save_json_vector(vec_list, path):
    """保存为 JSON 数组，每个元素为 3D 向量"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump([v.tolist() for v in vec_list], f, indent=2)

def barycentric_coordinates_tensor(p, a, b, c):
    """
    p, a, b, c: (..., 3)
    返回重心坐标 (..., 3)
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1)  # (..., 3)

def find_containing_triangle_batch(d, verts, faces):
    """
    d: (B, 3) 方向向量
    verts: (N, 3)
    faces: (M, 3)
    返回:
        face_ids: (B,)
        face_indices: (B, 3)
        bary: (B, 3)
    """

    # === Step 1: normalize d ===
    d = d / d.norm(dim=-1, keepdim=True)  # (B,3)

    # === Step 2: compute face normals ===
    v0 = verts[faces[:, 0]]  # (M,3)
    v1 = verts[faces[:, 1]]  # (M,3)
    v2 = verts[faces[:, 2]]  # (M,3)

    face_normals = torch.cross(v1 - v0, v2 - v0)   # (M,3)
    face_normals = face_normals / face_normals.norm(dim=-1, keepdim=True)

    # === Step 3: compute dot(d, normal) for all triangles ===
    # dot: (B, M)
    dots = torch.matmul(d, face_normals.T)

    # === Step 4: for each d, find the triangle with largest dot ===
    face_ids = torch.argmax(dots, dim=1)   # (B,)

    # === Step 5: gather corresponding triangle vertices ===
    tri = faces[face_ids]                  # (B,3)
    a = verts[tri[:, 0]]                   # (B,3)
    b = verts[tri[:, 1]]                   # (B,3)
    c = verts[tri[:, 2]]                   # (B,3)

    # === Step 6: compute barycentric coordinates ===
    bary = barycentric_coordinates_tensor(d, a, b, c)  # (B,3)

    return tri, bary



def interpolate_on_multi_sphere(points,               # [B,3] 任意3D点
                                verts_unit,           # [V,3] 单位球 geodesic mesh 顶点
                                faces,                # [M,3]
                                radii,                # [K]  不同球面半径(升序)
                                latent_per_sphere):   # [K,V,C] 每个球面的顶点latent
    """
    返回:
        feat: [B,C]  在最近两层球面上 + 重心 + 半径插值得到的特征
        tri:  [B,3]  使用的三角形顶点索引（在verts_unit上的）
        idx_lower, idx_upper: [B] 半径插值使用的两个球层索引
    """
    device = points.device
    radii = radii.to(device)              # [K]
    verts_unit = verts_unit.to(device)    # [V,3]
    faces = faces.to(device)              # [M,3]
    latent_per_sphere = latent_per_sphere.to(device)  # [K,V,C]

    B = points.shape[0]

    # --- 半径和单位方向 ---
    r = points.norm(dim=-1)                  # [B]
    d = points / r.unsqueeze(-1)             # [B,3]

    # --- 找到相邻两层球面 ---
    # radii 已排序
    idx_upper = torch.searchsorted(radii, r)        # [B]
    K = radii.shape[0]
    idx_upper = idx_upper.clamp(0, K - 1)
    idx_lower = (idx_upper - 1).clamp(0, K - 1)

    # --- 在单位球上找到包含三角形 + 重心坐标 ---
    # 你需要自行实现这个 batch 版本:
    # tri: [B,3], bary: [B,3]
    tri, bary = find_containing_triangle_batch(d, verts_unit, faces)

    u = bary[:, 0:1]   # [B,1]
    v = bary[:, 1:2]
    w = bary[:, 2:3]

    tri0 = tri[:, 0]   # [B]
    tri1 = tri[:, 1]
    tri2 = tri[:, 2]

    # --- 对两层球面分别做角度上的重心插值 ---
    C = latent_per_sphere.shape[-1]

    # flatten sphere index + vertex index 以便一次 gather
    # 方式1：直接循环两层（简单易懂）：
    def sample_on_sphere(k_idx):
        # k_idx: [B] sphere 索引
        # latent_per_sphere: [K,V,C]
        # 我们想得到 latent_per_sphere[k_idx, tri?, :]  => [B,C]
        # 用高级索引:
        feat0 = latent_per_sphere[k_idx, tri0, :]  # [B,C]
        feat1 = latent_per_sphere[k_idx, tri1, :]
        feat2 = latent_per_sphere[k_idx, tri2, :]
        feat  = feat0 * u + feat1 * v + feat2 * w  # [B,C]
        return feat

    feat0 = sample_on_sphere(idx_lower)    # [B,C]
    feat1 = sample_on_sphere(idx_upper)    # [B,C]

    # --- 半径方向插值 ---
    r0 = radii[idx_lower]   # [B]
    r1 = radii[idx_upper]   # [B]
    den = (r1 - r0).abs().clamp(min=1e-6)
    t = ((r - r0) / den).clamp(0, 1).unsqueeze(-1)   # [B,1]
    
    feat = feat0 * (1 - t) + feat1 * t              # [B,C]

    return feat, tri, idx_lower, idx_upper

def render(testbed,scene_path,output_path,object_data_list):
    testbed.load_scene(scene_path)
    scene = testbed.scene
    final_resolution = 512
    testbed.scene.camera.focalLength = 0
    testbed.scene.camera.nearPlane = 0.001
    
    testbed.scene.add_impostor()
    testbed.scene.impostor.loadFromFolder(testbed.device,r"H:\Falcor\scripts\python\scenes\onlybunny\level0")
    testbed.run()
    cnt = 0
    for i in range(len(object_data_list)):
        tex2 = testbed.capture_output(
             './{}_{}.exr'.format(object_data_list[i], cnt), i)
    cnt += 1
    print(1)
import pyexr

def nonzero_bbox(img):
    """
    输入: img -- numpy 数组, 形状可以是 (H, W) 或 (H, W, 1)
    输出: (ymin, ymax, xmin, xmax)
    """
    if img.ndim == 3:
        img = img[..., 0]  # 去掉最后一维

    # 找出所有非零坐标
    ys, xs = np.nonzero(img)

    if len(xs) == 0 or len(ys) == 0:
        return None  # 没有非零像素

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    return ymin, ymax, xmin, xmax
def generate_impostor_by_falcor(resolution,testbed,scene_path,output_path,object_data_dict,sellect_list,down=False,scale=False,scale_reference_path=None):
    testbed.load_scene(scene_path)
    scene = testbed.scene
    a = f3_to_numpy(scene.bounds.min_point)
    b = f3_to_numpy(scene.bounds.max_point)
    centor,object_radius,o_radius,p_radius = sampling_radii_from_aabb(a,b)
    final_resolution = resolution
    testbed.scene.camera.focalLength = 0
    testbed.scene.camera.nearPlane = 0.001
    testbed.scene.add_impostor()
    base = 1
    basic_info = {}
    basic_info["radius"] = o_radius
    basic_info["centorWS"] = centor.tolist()
    
    for subdiv_level in range(1,6):
        level_resolution = final_resolution
        testbed.resize_frame_buffer(level_resolution,level_resolution)
        basic_info["level"] = subdiv_level
        basic_info["texDim"] = [level_resolution,level_resolution]
        basic_info["invTexDim"] = [1/(level_resolution),1/(level_resolution)] 
        basic_info["baseCameraResolution"] = 2048

        level_output_path = output_path + "level{}/".format(subdiv_level)
        if not os.path.exists(level_output_path):
            os.makedirs(level_output_path)

        verts, faces = geodesic_impostor_mesh(subdiv_level)
        faces_list = faces.tolist()  
        #lookup_table = build_lookup_texture(verts,faces,resolution=128)
        lookup_table = build_lookup_texture_speedup_chunk(verts,faces,resolution=2048,chunk_size=512)
        #lookup_table2 = build_lookup_texture_speedup(verts,faces,resolution=256)
        lookup_uint = lookup_table.astype(np.uint16)
        cv2.imwrite(level_output_path + "lookup_uint16.png", lookup_uint)
        with open(level_output_path + "faces.json", "w") as f:
            json.dump(faces_list, f, indent=4)
        
        camera_positions = centor + verts * o_radius 
        # --- 统计 y 轴最大/最小值及索引 ---
        y_values = camera_positions[:, 1]  # y 分量
        y_min_idx = np.argmin(y_values)
        y_max_idx = np.argmax(y_values)
        y_min = y_values[y_min_idx]
        y_max = y_values[y_max_idx]
        r_list = []
        f_list = []
        u_list = []
        p_list = []
        print(f"[Level {subdiv_level}] y_min = {y_min:.4f} (index {y_min_idx}),  "
            f"y_max = {y_max:.4f} (index {y_max_idx})")
        
        cnt = 0
        radius_info=[]
        for single_pos in camera_positions:
            testbed.scene.camera.position = single_pos
            if scale and scale_reference_path is not None:
                rd = pyexr.read(scale_reference_path + "/level{}/depth_{:05d}.exr".format(subdiv_level,cnt))[...,:1]
                ymin,ymax,xmin,xmax = nonzero_bbox(rd)
                scale = (level_resolution // 2) / (max(level_resolution // 2 - ymin,ymax - level_resolution // 2,level_resolution // 2 - xmin,xmax - level_resolution // 2) + 1)
                scale = max(scale,1)
                scale_radius = o_radius *scale
            else:
                scale_radius = o_radius * (9-scale)/8
            radius_info.append(scale_radius)
            print(scale_radius)
            testbed.scene.camera.target = normalize(centor - single_pos) * scale_radius * 2 + single_pos
            up = unity_style_up(normalize(centor - single_pos))
            r,u,f = compute_camera_basis(single_pos,normalize(centor - single_pos) * scale_radius * 2 + single_pos,up)
            r_list.append(r)
            u_list.append(u)
            f_list.append(f)
            p_list.append(single_pos)
            testbed.scene.camera.up = up
            testbed.run()

            for name in sellect_list:
                index = object_data_dict[name]
                tex2 = testbed.capture_output(
                    level_output_path + '{}_{:05d}.exr'.format(name, cnt), index)
            cnt += 1
            print(subdiv_level,cnt)
            #print("gg")
        right_path = os.path.join(level_output_path, "right.json")
        with open(right_path, "w") as f:
            json.dump([r.tolist() for r in r_list], f, indent=4)

        # === 保存 up ===
        up_path = os.path.join(level_output_path, "up.json")
        with open(up_path, "w") as f:
            json.dump([u.tolist() for u in u_list], f, indent=4)

        # === 保存 forward ===
        forward_path = os.path.join(level_output_path, "forward.json")
        with open(forward_path, "w") as f:
            json.dump([f_.tolist() for f_ in f_list], f, indent=4)
        radius_path = os.path.join(level_output_path, "radius.json")
        with open(radius_path, "w") as f:
            json.dump(radius_info, f, indent=4)
        position_path = os.path.join(level_output_path, "position.json")
        with open(position_path, "w") as f:
            json.dump([p_.tolist() for p_ in p_list], f, indent=4)
        with open(level_output_path + "basic_info.json", "w") as f:
            json.dump(basic_info, f, indent=4)
        print(f"✅ Saved right/up/forward/position JSONs to {level_output_path}")
        if down:
            base = base * 2
            


import torch
import torch.nn.functional as F
#["albedo","specular","depth","emissive","normal","position","roughness"]
buffer_channel ={
    "albedo":3,
    "specular":3,
    "depth":1,
    "emission":3,
    "normal":3,
    "position":3,
    "roughness":1,
    "view":3,
    "raypos":3,
    "AccumulatePassoutput":3,
    "mind":1,
    "reflect":3,
    "idepth0":3,
    "idepth1":3,
    "idepth2":3,
    "idirection0":4,
    "idirection1":4,
    "idirection2":4,
    "uv0":4,
    "uv1":4,
    "uv2":4,
}
class ImpostorPT:
    def __init__(self):
        # scalar
        self.radius = None       # float
        self.centerWS = None     # (3,)

        # textures
        # texDepth:  (H, W, V, 1)
        # texAlbedo: (H, W, V, 4)
        self.texDict = {}
        self.texFaceIndex = None     # (H, W)
        
        # camera triplets (V, 3)
        self.cFace = None

        # camera params for all V
        # (V, 3)
        self.cPosition = None
        self.cRight    = None
        self.cUp       = None
        self.cForward  = None
        self.cRadius  = None

def normalize_tensor(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def dot(a, b):
    return (a * b).sum(dim=-1, keepdim=True)

def saturate(x):
    return x.clamp(0.0, 1.0)


def OctEncode(n):   # n: (B,3)
    denom = torch.abs(n).sum(dim=-1, keepdim=True)
    n = n / (denom + 1e-8)

    neg = (n[:,2] < 0).float().unsqueeze(-1)
    xy = n[:,:2]

    # flip xy when z<0
    flipped = (1 - torch.abs(xy[:,[1,0]])) * torch.sign(xy)
    xy = xy * (1-neg) + flipped * neg
    
    return xy * 0.5 + 0.5   # (B,2)
def ray_sphere_intersect(rayPos, rayDir, center, radius):
    """
    输入:
    rayPos: (B,3)
    rayDir: (B,3)
    center: (1,3)
    radius: float

    返回:
    hit: (B,) 是否命中
    tStart: (B,) 进入球的 t
    """

    L = rayPos - center   # (B,3)
    b = dot(L, rayDir)[:,0]
    c = (L * L).sum(dim=-1) - radius * radius

    # 判别式
    disc = b*b - c
    hit = disc >= 0

    t0 = -b - torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = -b + torch.sqrt(torch.clamp(disc, min=0.0))
    t0 = torch.where(hit, t0, torch.zeros_like(t0))
    t0 = torch.where(t0<0, 0, t0)
    hit = hit & (t1 >= 0)
    return hit, t0

def getNearestImpostorView(impostor, direction):
    """
    direction: (B,3)
    return: (B,3) 三个视图 index
    """
    uv = OctEncode(direction)     # (B,2)

    H, W = impostor.texFaceIndex.shape

    # uv → pixel
    x = (uv[:,0] * (W-1)).long().clamp(0, W-1)
    y = (uv[:,1] * (H-1)).long().clamp(0, H-1)

    viewIdx = impostor.texFaceIndex[y, x]        # (B,)
    tri = impostor.cFace[viewIdx]                # (B,3)

    return tri

def sampleImpostorPT(impostor, rayDirW, rayPosW):
    """
    rayDirW: (B,3)
    rayPosW: (B,3)
    return:  (B,4)
    """

    B = rayDirW.shape[0]
    device = rayDirW.device

    rayDirW = normalize_tensor(rayDirW)

    # ---------------------------
    # 平面 intersection
    # ---------------------------
    center = impostor.centerWS.to(device).unsqueeze(0)      # (1,3)
    planeNormal = normalize_tensor(center - rayPosW)               # (B,3)

    denom = dot(rayDirW, planeNormal)[:,0]                  # (B,)
    # 平行的返回零
    parallel = (denom.abs() < 1e-5)

    t = dot(center - rayPosW, planeNormal)[:,0] / (denom + 1e-8)
    hitPosW = rayPosW + rayDirW * t.unsqueeze(-1)           # (B,3)

    # ---------------------------
    # 查询最近的 3 个视图
    # ---------------------------
    triIdx = getNearestImpostorView(impostor, -planeNormal)  # (B,3)

    invRadius2 = 0.5 / impostor.radius

    H, W, V, _ = impostor.texDepth.shape

    # 输出
    out = torch.zeros((B,4), device=device)

    # 三个结果
    samples = torch.zeros((B,3,4), device=device)
    weights = torch.zeros((B,3), device=device)

    sigma = 0.2

    # ---------------------------
    # for i in 0..2 (视图)
    # ---------------------------
    for i in range(3):
        viewIdx = triIdx[:,i]                    # (B,)
        # gather camera params
        camPos = impostor.cPosition[viewIdx]     # (B,3)
        camR   = impostor.cRight[viewIdx]
        camU   = impostor.cUp[viewIdx]
        camF   = impostor.cForward[viewIdx]

        delta = hitPosW - camPos

        x = dot(delta, camR)[:,0]                # (B,)
        y = dot(delta, camU)[:,0]

        uv = torch.stack([x, y], dim=-1) * invRadius2 + 0.5   # (B,2)

        # 边界 mask
        valid_uv = (uv[:,0] >= 0) & (uv[:,0] <= 1) & \
                   (uv[:,1] >= 0) & (uv[:,1] <= 1)

        # uv → pixel
        px = (uv[:,0] * (W-1)).long().clamp(0, W-1)
        py = (uv[:,1] * (H-1)).long().clamp(0, H-1)

        # ---------------------------
        # 采样 depth
        # texDepth: (H,W,V,1)
        # ---------------------------
        depth = impostor.texDepth[py, px, viewIdx, 0]    # (B,)
        valid_depth = (depth > 0)

        valid = valid_uv & valid_depth

        # ---------------------------
        # corrected_depth
        # ---------------------------
        P_view = camPos + camR * x.unsqueeze(-1) \
                        + camU * y.unsqueeze(-1) \
                        + camF * depth.unsqueeze(-1)

        vec_ray = P_view - rayPosW
        corrected_depth = dot(vec_ray, rayDirW)[:,0]

        # ---------------------------
        # sample albedo (float4)
        # texAlbedo: (H,W,V,4)
        # ---------------------------
        albedo = impostor.texAlbedo[py, px, viewIdx]      # (B,4)

        # override
        albedo2 = albedo.clone()
        albedo2[:,3] = corrected_depth
        albedo2[:,:3] = P_view

        samples[:,i] = albedo2 * valid.unsqueeze(-1)

        # ---------------------------
        # weight = exp(-theta^2 / (2σ^2))
        # ---------------------------
        cam_to_p = P_view - camPos
        cos_theta = saturate((normalize_tensor(cam_to_p) * rayDirW).sum(-1))  # (B,)
        theta = torch.acos(cos_theta)

        w = torch.exp(-(theta * theta) / (2 * sigma * sigma))
        w = w * valid.float()

        weights[:,i] = w

    totalWeight = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    w_norm = weights / totalWeight

    blended = (samples * w_norm.unsqueeze(-1)).sum(dim=1)

    # 平行 ray 的结果设为 0
    blended[parallel] = 0

    return blended

def project_world_to_view_uv(impostor, posW, viewIdx, device="cuda"):
    """
    posW: (B,3)
    viewIdx: (B,) long
    返回:
        uv_norm: (B,2) 归一化到 [-1, 1]
    """

    camPos = impostor.cPosition[viewIdx].to(device)   # (B,3)
    camR   = impostor.cRight[viewIdx].to(device)      # (B,3)
    camU   = impostor.cUp[viewIdx].to(device)         # (B,3)
    camF   = impostor.cForward[viewIdx].to(device)    # (B,3)
    radius = impostor.cRadius[viewIdx].to(device)     # (B,)

    delta = posW - camPos                             # (B,3)

    # 与 sampleImpostorAccuratePT 相同投影公式
    x = (delta * camR).sum(-1)                         # (B,)
    y = (delta * camU).sum(-1)

    uv = torch.stack([x, y], dim=-1) * (0.5 / radius) + 0.5  # [0,1]

    # 转换到 [-1,1]
    uv_norm = uv * 2.0 - 1.0

    return uv_norm

def sampleImpostorAccuratePT(
    impostor,                          # Impostor 数据结构
    rayDirW, rayPosW,                  # (B,3)
    N=512,
    W=512,
    H=512,
    debug=False,
    reference_pos = None
):
    device = rayDirW.device
    B = rayDirW.shape[0]

    rayDirW = normalize_tensor(rayDirW)

    center = impostor.centerWS.to(device).unsqueeze(0)  # (1,3)




    # 最近 3 个视图
    triCameraIdx = getNearestImpostorView(impostor, -rayDirW)   # (B,3)
    if triCameraIdx.shape[0] == 64 * 64:
        pyexr.write(r"./triCameraIdx.exr",triCameraIdx.reshape(64,64,3).cpu().numpy())
    # 结果存储
    samples    = torch.zeros((B,3,4), device=device)   # 每个视图的 RGB 和 placeholder
    samples_position    = torch.zeros((B,3,3), device=device)   # 每个视图的 RGB 和 placeholder
    weights    = torch.zeros((B,3), device=device)
    minTheta   = torch.full((B,), 1e10, device=device)
    final_pos = torch.zeros((B,3), dtype=torch.float32, device=device)
    final_emission = torch.zeros((B,3), dtype=torch.float32, device=device)
    intersect_emission = torch.zeros((B,3,3), dtype=torch.float32, device=device)
    triCameraP = torch.zeros((B,3,3), dtype=torch.float32, device=device)
    triCameraU = torch.zeros((B,3,3), dtype=torch.float32, device=device)
    triCameraR = torch.zeros((B,3,3), dtype=torch.float32, device=device)
    triSellect = torch.zeros((B,3,3), dtype=torch.long, device=device)

    final_idx  = torch.zeros((B,), dtype=torch.long, device=device)

    sigma = 0.5

    # -------------------------
    # Ray / sphere intersection
    # -------------------------
    hitSphere, tStart = ray_sphere_intersect(rayPosW, rayDirW, center, impostor.radius)
    tStart = tStart.unsqueeze(-1)
    searchStart = rayPosW + rayDirW * tStart              # (B,3)

    # 所有 step 的采样点: (B, N, 3)
    steps = (torch.arange(N, device=device).float() + 0.5) * (2.0 * impostor.radius / N)
    steps = steps.view(1,N,1)                             # (1,N,1)
    samplePos = searchStart.unsqueeze(1) + rayDirW.unsqueeze(1) * steps   # (B,N,3)
    rayDepth = tStart + steps[:,:,0]
    # -------------------------
    # 遍历 3 个视图，但内部完全向量化 (B,N)
    # -------------------------
    hitFoundFinal = torch.zeros((B,3)).cuda()
    uv_back = torch.zeros((B,3,2)).cuda()
    if reference_pos is not None:
        final_pos = reference_pos
        proj_uv = []
        proj_viewIdx = triCameraIdx  # (B,3)

        for i in range(3):
            uv_i = project_world_to_view_uv(
                impostor,
                final_pos,                     # (B,3)
                triCameraIdx[:, i],            # (B,)
                device=device
            )   # (B,2)

            proj_uv.append(uv_i)

        proj_uv = torch.stack(proj_uv, dim=1)   # (B,3,2)
        uvi = torch.cat([proj_uv,proj_viewIdx.unsqueeze(-1).float()],dim=-1)
        return uvi, final_pos
    for i in range(3):
        viewIdx = triCameraIdx[:,i]               # (B,)
        
        camPos = impostor.cPosition[viewIdx]      # (B,3)
        camR   = impostor.cRight[viewIdx]
        camU   = impostor.cUp[viewIdx]
        camF   = impostor.cForward[viewIdx]
        viewRadius = impostor.cRadius[viewIdx]
        triCameraP[:,i,:] = camPos
        triCameraU[:,i,:] = camU
        triCameraR[:,i,:] = camR
        # (B,N,3)
        delta = samplePos - camPos.unsqueeze(1)

        x = (delta * camR.unsqueeze(1)).sum(-1)    # (B,N)
        y = (delta * camU.unsqueeze(1)).sum(-1)    # (B,N)

        uv = torch.stack([x, y], dim=-1) * (0.5 / viewRadius.unsqueeze(-1)) + 0.5
        if debug:
            pyexr.write(r"H:\Falcor\media\inv_rendering_scenes\object_level_config\Bunny/uv_{}.exr".format(viewIdx[0].item()),uv[:,0,:].reshape(W,H,2).cpu().numpy())
            pyexr.write(r"H:\Falcor\media\inv_rendering_scenes\object_level_config\Bunny/uv100_{}.exr".format(viewIdx[0].item()),uv[:,100,:].reshape(W,H,2).cpu().numpy())
        valid_uv = (uv[...,0]>=0)&(uv[...,0]<=1)&(uv[...,1]>=0)&(uv[...,1]<=1)
        uv_back[:,i,:] = uv[:,0,:]
        H,W,V,_ = impostor.texDict["depth"].shape

        # 量化 uv 到像素
        px = (uv[...,0] * (W)).long().clamp(0, W-1)
        py = (uv[...,1] * (H)).long().clamp(0, H-1)

        # --------------------------
        # depth 采样 (B,N)
        # --------------------------
        lenth_delta = delta.norm(dim=-1)
        depth = impostor.texDict["depth"][py, px, viewIdx[:,None], 0]   # (B,N)

        if debug:
            pyexr.write(r"H:\Falcor\media\inv_rendering_scenes\object_level_config\Bunny/pxy_{}.exr".format(viewIdx[0].item()),pxy[:,0,:].reshape(W,H,2).cpu().numpy())
            pyexr.write(r"H:\Falcor\media\inv_rendering_scenes\object_level_config\Bunny/pxy300_{}.exr".format(viewIdx[0].item()),pxy[:,300,:].reshape(W,H,2).cpu().numpy())
        P_view = (
            camPos.unsqueeze(1) +
            camR.unsqueeze(1) * x.unsqueeze(-1) +
            camU.unsqueeze(1) * y.unsqueeze(-1) +
            camF.unsqueeze(1) * depth.unsqueeze(-1)
        ) 
        hitPosW = samplePos[torch.arange(B), :]   # (B,3)
        hitPosV = P_view[torch.arange(B), :]   # (B,3)

        theta = (hitPosV - hitPosW).norm(dim=-1)
        # 条件：uv valid、depth valid（深度非零）、并且 rayDepth >= depth
        depth_valid = depth < 5
    
        intersect_mask = valid_uv & depth_valid & (lenth_delta >= depth) & (theta < (viewRadius/256.0)) 

        # 现在 intersect_mask: (B,N) = True at intersection positions

        # 为了找到“第一个 True”，把 False 设成一个极大值，把 True 的 index 保留
        # idx_tensor[i,j] = j if mask True else large number
        
        indices = torch.arange(N, device=device).view(1, N).expand(B, N)
        indices_masked = torch.where(intersect_mask, indices, torch.full_like(indices, N))

        # 取每条 ray 的最小 index，即 first hit
        first_hit_idx = indices_masked.min(dim=1).values  # (B,)

        # hitFound: 如果 first_hit_idx < N 就说明有命中
        hitFound = first_hit_idx < N 
    

        safe_idx = torch.clamp(first_hit_idx, 0, N - 1)        # 防止 miss 时越界
        sellect_depth = rayDepth[torch.arange(B), safe_idx]
        hitPosW = hitPosW[torch.arange(B), safe_idx]
        sellect_y = py[torch.arange(B), safe_idx]
        sellect_x = px[torch.arange(B), safe_idx]
        emission = impostor.texDict["emission"][sellect_y, sellect_x, viewIdx[:], :] 
        
        
        intersect_emission[:,i,:] = emission
        hitFoundFinal[:,i] = hitFound.float()
        final_pos = torch.where((sellect_depth.unsqueeze(-1) < minTheta.unsqueeze(-1)) & hitSphere.unsqueeze(-1) & hitFound.unsqueeze(-1), hitPosW, final_pos)
        final_emission = torch.where((sellect_depth.unsqueeze(-1) < minTheta.unsqueeze(-1)) & hitSphere.unsqueeze(-1) & hitFound.unsqueeze(-1), emission, final_emission)
        minTheta = torch.where((sellect_depth<minTheta) &hitSphere & hitFound ,sellect_depth,minTheta)
    any_hit = minTheta < 1e9   # 或者用 hitFoundFinal.any(dim=1)

    final_pos = torch.where(
        any_hit.unsqueeze(-1),
        final_pos,
        torch.zeros_like(final_pos)
    )

    final_emission = torch.where(
        any_hit.unsqueeze(-1),
        final_emission,
        torch.zeros_like(final_emission)
    )
    
    if debug:
        return final_pos,final_emission,intersect_emission,triCameraIdx,searchStart,triCameraP,triCameraU,triCameraR,triSellect
    else:
        proj_uv = []
        proj_viewIdx = triCameraIdx  # (B,3)

        for i in range(3):
            uv_i = project_world_to_view_uv(
                impostor,
                final_pos,                     # (B,3)
                triCameraIdx[:, i],            # (B,)
                device=device
            )   # (B,2)

            proj_uv.append(uv_i)

        proj_uv = torch.stack(proj_uv, dim=1)   # (B,3,2)
        uvi = torch.cat([proj_uv,proj_viewIdx.unsqueeze(-1).float()],dim=-1)
        return uvi, final_pos

def sampleImpostorDebugPixel(
    impostor,
    rayDirW, rayPosW,    # (1,3)
    viewIdx,             # 手动指定视图 idx
    N=512,
    device="cuda"
):
    """
    返回:
        debug_step_img  : step-based brightness
        debug_depth_img : depth difference visualization
        debug_theta_img : theta visualization
        hit_pixel       : (px,py) or None
    """
    rayDirW = torch.nn.functional.normalize(rayDirW, dim=-1)
    print(rayDirW)
    device = rayDirW.device

    # ---------------------
    # camera info
    # ---------------------
    camPos = impostor.cPosition[viewIdx:viewIdx+1].to(device)
    camR   = impostor.cRight[viewIdx:viewIdx+1].to(device)
    camU   = impostor.cUp[viewIdx:viewIdx+1].to(device)
    camF   = impostor.cForward[viewIdx:viewIdx+1].to(device)
    viewRadius = impostor.cRadius[viewIdx].to(device)

    center = impostor.centerWS.to(device).unsqueeze(0)
    print(center)

    # ---------------------
    # sphere intersection
    # ---------------------
    hitSphere, tStart = ray_sphere_intersect(rayPosW, rayDirW, center, impostor.radius)
    tStart = tStart.unsqueeze(-1)

    searchStart = rayPosW + rayDirW * tStart

    # marching steps
    steps = (torch.arange(N, device=device).float() + 0.5) * (2.0 * impostor.radius / N)
    steps = steps.view(1, N, 1)
    samplePos = searchStart.unsqueeze(1) + rayDirW.unsqueeze(1) * steps   # (1,N,3)

    # ---------------------
    # 投影
    # ---------------------
    delta = samplePos - camPos.unsqueeze(1)

    x = (delta * camR.unsqueeze(1)).sum(-1)
    y = (delta * camU.unsqueeze(1)).sum(-1)

    uv = torch.stack([x, y], dim=-1) * (0.5 / viewRadius) + 0.5

    valid_uv = (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & \
               (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)

    H, W, V, _ = impostor.texDict["depth"].shape
    px = (uv[..., 0] * W).long().clamp(0, W-1)
    py = (uv[..., 1] * H).long().clamp(0, H-1)

    # ---------------------
    # depth 查询
    # ---------------------
    depth = impostor.texDict["depth"][py, px, viewIdx, 0]  # (1,N)

    P_view = (
        camPos.unsqueeze(1) +
        camR.unsqueeze(1) * x.unsqueeze(-1) +
        camU.unsqueeze(1) * y.unsqueeze(-1) +
        camF.unsqueeze(1) * depth.unsqueeze(-1)
    )

    hitPosW_all = samplePos[0]     # (N,3)
    hitPosV_all = P_view[0]        # (N,3)

    theta = (hitPosV_all - hitPosW_all).norm(dim=-1)  # (N,)

    # ---------------------
    # 计算 intersect mask
    # ---------------------
    rayDepth = (tStart + steps[:, :, 0])[0]
    lenth_delta = delta.norm(dim=-1)[0]
    depth_valid = depth[0] > 1e-6
    uv_valid = valid_uv[0]

    intersect_mask = uv_valid & depth_valid & (lenth_delta >= depth[0]) & (theta < viewRadius / 128.0)

    hit_pixel = None
    if intersect_mask.any():
        idx = torch.nonzero(intersect_mask)[0, 0].item()
        hx, hy = px[0, idx].item(), py[0, idx].item()
        hit_pixel = (hx, hy)

    # -----------------------------------------------------
    # Debug 图像构建
    # -----------------------------------------------------

    debug_step_img  = torch.zeros((H, W, 3), device=device)
    debug_depth_img = torch.zeros((H, W, 3), device=device)
    debug_theta_img = torch.zeros((H, W, 3), device=device)

    # --- Step color (灰度从暗到亮) ---
    brightness = torch.linspace(0.1, 1.0, N, device=device)
    step_colors = torch.stack([brightness, brightness, brightness], dim=-1)

    # --- Depth diff color ---
    depth_diff = lenth_delta - depth[0]
    depth_norm = depth_diff / (depth_diff.max() + 1e-6)
    depth_colors = torch.stack([depth_norm, depth_norm, depth_norm], dim=-1)

    # --- Theta color ---
    theta_norm = theta 
    theta_colors = torch.stack([theta_norm, theta_norm, theta_norm], dim=-1)
    print(viewRadius / 128.0)
    # 画采样点
    for i in range(N):
        yy = py[0, i]
        xx = px[0, i]

        debug_step_img[yy, xx]  = step_colors[i]
        debug_depth_img[yy, xx] = depth_colors[i]
        debug_theta_img[yy, xx] = theta_colors[i]

    # 命中点画红色
    if hit_pixel is not None:
        hx, hy = hit_pixel
        red = torch.tensor([1.0, 0.0, 0.0], device=device)
        debug_step_img[hy, hx]  = red
        debug_depth_img[hy, hx] = red
        debug_theta_img[hy, hx] = red

    return debug_step_img, debug_depth_img, debug_theta_img




def load_impostor(name,level):
    path = r"H:\Falcor\media\inv_rendering_scenes\object_level_config_test/{}level{}/".format(name,level)
    impostor = ImpostorPT()
    with open(path + "basic_info.json","r") as file:
        basic_info = json.load(file)
    impostor.radius = basic_info["radius"]
    impostor.centerWS = torch.tensor(basic_info["centorWS"], dtype=torch.float32).cuda()
    with open(path + "faces.json","r") as file:
        faces_info = json.load(file)
    impostor.cFace = torch.tensor(faces_info, dtype=torch.int32).cuda()
    with open(path + "forward.json","r") as file:
        forward_info = json.load(file)
    impostor.cForward = torch.tensor(forward_info, dtype=torch.float32).cuda()
    with open(path + "up.json","r") as file:
        up_info = json.load(file)
    impostor.cUp = torch.tensor(up_info, dtype=torch.float32).cuda()
    with open(path + "right.json","r") as file:
        right_info = json.load(file)
    impostor.cRight = torch.tensor(right_info, dtype=torch.float32).cuda()
    with open(path + "position.json","r") as file:
        position_info = json.load(file)
    impostor.cPosition = torch.tensor(position_info, dtype=torch.float32).cuda()
    with open(path + "radius.json","r") as file:
        radius_info = json.load(file)
    impostor.cRadius = torch.tensor(radius_info, dtype=torch.float32).cuda().unsqueeze(-1)
    # 读取 lookup_uint16.png,用opencv读取
    texFaceIndex = cv2.imread(path + "lookup_uint16.png", cv2.IMREAD_UNCHANGED)
    impostor.texFaceIndex = torch.tensor(texFaceIndex, dtype=torch.int32).cuda()
    subdirs = [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]
    impostor.texDict = {}
    for key in ["albedo","specular","depth","emission","normal","position","roughness","view","raypos"]:
        impostor.texDict[key] = torch.zeros((512,512,42,buffer_channel[key])).cuda()
    for file in subdirs:
        for name in ["albedo","specular","depth","emission","normal","position","roughness","view","raypos"]:
            id = int(file)
            
            image_path = path + file + "/" + name + f"_{id:05d}.exr"
            impostor.texDict[name][:,:,id,:] = torch.Tensor(pyexr.read(image_path)[:,:,:buffer_channel[name]]).cuda()

    return impostor

def load_impostor2(name,level):
    path = r"H:\Falcor\media\inv_rendering_scenes\object_level_config_test/{}level{}/".format(name,level)
    impostor = ImpostorPT()
    with open(path + "basic_info.json","r") as file:
        basic_info = json.load(file)
    impostor.radius = basic_info["radius"]
    impostor.centerWS = torch.tensor(basic_info["centorWS"], dtype=torch.float32).cuda()
    with open(path + "faces.json","r") as file:
        faces_info = json.load(file)
    impostor.cFace = torch.tensor(faces_info, dtype=torch.int32).cuda()
    with open(path + "forward.json","r") as file:
        forward_info = json.load(file)
    impostor.cForward = torch.tensor(forward_info, dtype=torch.float32).cuda()
    with open(path + "up.json","r") as file:
        up_info = json.load(file)
    impostor.cUp = torch.tensor(up_info, dtype=torch.float32).cuda()
    with open(path + "right.json","r") as file:
        right_info = json.load(file)
    impostor.cRight = torch.tensor(right_info, dtype=torch.float32).cuda()
    with open(path + "position.json","r") as file:
        position_info = json.load(file)
    impostor.cPosition = torch.tensor(position_info, dtype=torch.float32).cuda()
    with open(path + "radius.json","r") as file:
        radius_info = json.load(file)
    impostor.cRadius = torch.tensor(radius_info, dtype=torch.float32).cuda().unsqueeze(-1)
    # 读取 lookup_uint16.png,用opencv读取
    texFaceIndex = cv2.imread(path + "lookup_uint16.png", cv2.IMREAD_UNCHANGED)
    impostor.texFaceIndex = torch.tensor(texFaceIndex, dtype=torch.int32).cuda()
    file_list = os.listdir(path)
    impostor.texDict = {}
    for key in ["albedo","specular","depth","emission","normal","position","roughness","view","raypos"]:
        impostor.texDict[key] = torch.zeros((42,512,512,buffer_channel[key])).cuda()
    subdirs = [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]
    for file in subdirs:
        for name in ["albedo","specular","depth","emission","normal","position","roughness","view","raypos"]:
            id = int(file)
            
            image_path = path + file + "/" + name + f"_{id:05d}.exr"
            impostor.texDict[name][id,:,:,:] = torch.Tensor(pyexr.read(image_path)[:,:,:buffer_channel[name]]).cuda()
    return impostor
def invert_matrix_4x4(M):
    return torch.inverse(M)

def transform_rays_world_to_local(rayPosW, rayDirW, M, device="cuda"):
    """
    rayPosW: (N,3)
    rayDirW: (N,3)
    M: (4,4)   world → object transform matrix
    返回:
        rayPos_local, rayDir_local
    """

    # 计算逆矩阵：object_local → world
    M_inv = torch.inverse(M).to(device)
  
    # -----------------------------
    # 1. 变换射线起点（齐次坐标）
    # -----------------------------
    ones = torch.ones((rayPosW.shape[0], 1), device=device)
    pos_homo = torch.cat([rayPosW, ones], dim=-1)         # (N,4)

    pos_local_homo = (M_inv @ pos_homo.T).T               # (N,4)
    rayPos_local = pos_local_homo[:, :3] / pos_local_homo[:, 3:4]

    # -----------------------------
    # 2. 变换方向（不包含平移）
    # -----------------------------
    R_inv = M_inv[:3, :3]                                 # rotation + scale inverse
    rayDir_local = (rayDirW @ R_inv.T)
    rayDir_local = torch.nn.functional.normalize(rayDir_local, dim=-1)

    return rayPos_local, rayDir_local


def transform_points_local_to_world(points_local, M, device="cuda"):
    """
    points_local: (N,3)
    M: (4,4) local → world transform matrix
    return: points_world (N,3)
    """
    ones = torch.ones((points_local.shape[0], 1), device=device)
    p_homo = torch.cat([points_local, ones], dim=-1)   # (N,4)

    p_world_homo = (M @ p_homo.T).T
    points_world = p_world_homo[:, :3] / p_world_homo[:, 3:4]

    return points_world



def compute_reflection_direction(rayDirW, normal):
    """
    输入:
        rayDirW: (N,3) 世界空间入射方向（已归一化）
        normal:  (N,3) 世界空间法线（必须为 unit vector）
    
    输出:
        reflectDirW: (N,3) 世界空间反射方向（已归一化）
    """

    # 保证 normal 是 unit-normal
    n = torch.nn.functional.normalize(normal, dim=-1)

    # d·n
    dn = (rayDirW * n).sum(dim=-1, keepdim=True)

    # 反射公式 r = d - 2(d·n)n
    reflectDir = rayDirW - 2 * dn * n

    # 归一化
    reflectDir = torch.nn.functional.normalize(reflectDir, dim=-1)

    return reflectDir




def chunk_process_debug(impostor,rayDir_local, rayPos_local,W,H,chunk=32768):
    """
    将射线拆分为多个批次执行 sampleImpostorAccuratePT
    rayDir_local: (N,3)
    rayPos_local: (N,3)
    返回: final_pos(N,3), final_emission(N,3)
    """
    N = rayDir_local.shape[0]
    out_pos = []
    out_emi = []
    out_ie = []
    out_ca = []
    out_pa = []
    out_cap = []
    out_cau = []
    out_car = []
    out_y = []
    out_x = []
    out_view = []
    for i in range(0, N, chunk):
        rd = rayDir_local[i:i+chunk]
        rp = rayPos_local[i:i+chunk]

        fp, fe, ie, ca,start_pos,cap,cau,car,oy = sampleImpostorAccuratePT(
            impostor,
            rd,
            rp,
            512, W, H
        )
        out_pos.append(fp)
        out_emi.append(fe)
        out_ie.append(ie)
        out_ca.append(ca)
        out_pa.append(start_pos)
        out_cap.append(cap)
        out_cau.append(cau)
        out_car.append(car)
        out_y.append(oy)
   
    return torch.cat(out_pos, dim=0), torch.cat(out_emi, dim=0), torch.cat(out_ie, dim=0), torch.cat(out_ca, dim=0), torch.cat(out_pa, dim=0), torch.cat(out_cap, dim=0), torch.cat(out_cau, dim=0), torch.cat(out_car, dim=0), torch.cat(out_y, dim=0)



def chunk_process(impostor,rayDir_local, rayPos_local, W,H,chunk=32768,reference_pos=None):
    """
    将射线拆分为多个批次执行 sampleImpostorAccuratePT
    rayDir_local: (N,3)
    rayPos_local: (N,3)
    返回: final_pos(N,3), final_emission(N,3)
    """
    N = rayDir_local.shape[0]
    out_uvi = []
    final_pos = []
    for i in range(0, N, chunk):
        rd = rayDir_local[i:i+chunk]
        rp = rayPos_local[i:i+chunk]
        if reference_pos != None:
            referp = reference_pos[i:i+chunk]
        else:
            referp = None
        uvi,pos = sampleImpostorAccuratePT(
            impostor,
            rd,
            rp,
            512, W, H,
            reference_pos=referp
        )
        out_uvi.append(uvi)
        final_pos.append(pos)
    return torch.cat(out_uvi, dim=0),torch.cat(final_pos, dim=0)

def sample_impostor_uv_grid(impostor, coords, tex_name="emission", device="cuda"):
    """
    使用 grid_sample 对 impostor 数据进行双线性采样。

    输入:
        coords: (B,3)
            coords[:,0] = u ∈ [-1,1]
            coords[:,1] = v ∈ [-1,1]
            coords[:,2] = view index (long)
        impostor.texDict[tex_name]: (H, W, V, C)

    输出:
        sampled: (B, C)  # 双线性采样结果
    """

    coords = coords.to(device)
    u = coords[:, 0]
    v = coords[:, 1]
    viewIdx = coords[:, 2].long()

    tex = impostor.texDict[tex_name].to(device)   # (H, W, V, C)
    H, W, V, C = tex.shape

    B = coords.shape[0]

    # ----------------------------------------------------
    # Step 1: 选出每个 sample 对应的视角纹理 (B, H, W, C)
    # ----------------------------------------------------
    # tex:     (H, W, V, C)
    # we want: (B, H, W, C)
    tex_B = tex[:, :, viewIdx, :]                  # (H, W, B, C)
    tex_B = tex_B.permute(2, 3, 0, 1)              # → (B, C, H, W)

    # ----------------------------------------------------
    # Step 2: 构建 grid_sample 的 grid
    # grid shape 必须是 (B, 1, 1, 2)
    # ----------------------------------------------------
    grid = torch.stack([u, v], dim=-1).view(B, 1, 1, 2)
    # grid 内必须保证 ∈ [-1,1]，你已经保证了
    grid = grid.clamp(-1, 1)

    # ----------------------------------------------------
    # Step 3: grid_sample 双线性采样
    # ----------------------------------------------------
    sampled = F.grid_sample(
        tex_B,          # (B,C,H,W)
        grid,           # (B,1,1,2)
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # sampled shape = (B, C, 1, 1) → squeeze
    return sampled.squeeze(-1).squeeze(-1)


def sample_impostor_uv_bilinear_byidx(coords, tex, device="cuda"):
    """
    手动 bilinear 从 impostor 采样：
    输入:
        coords: (B,3)  -> (u,v,viewIdx)
            u,v ∈ [-1,1]
            viewIdx: long

    输出:
        sampled: (B,C)
    """

    coords = coords.to(device)
    u = coords[:, 0]          # [-1,1]
    v = coords[:, 1]
    viewIdx = coords[:, 2].long()

    
    H, W, V, C = tex.shape

    # -------------------------
    # 将 [-1,1] 映射到 [0,W-1] / [0,H-1]
    # -------------------------
    uf = (u + 1) * 0.5 * (W - 1)     # float pixel space
    vf = (v + 1) * 0.5 * (H - 1)

    # 找到邻域像素
    x0 = uf.floor().long().clamp(0, W - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y0 = vf.floor().long().clamp(0, H - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    # bilinear 权重
    wx = uf - x0.float()   # (B,)
    wy = vf - y0.float()

    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy

    # -------------------------
    # 逐视角索引采样：VERY MEMORY EFFICIENT
    # -------------------------
    # tex: (H, W, V, C)
    s00 = tex[y0, x0, viewIdx, :]   # (B,C)
    s01 = tex[y1, x0, viewIdx, :]
    s10 = tex[y0, x1, viewIdx, :]
    s11 = tex[y1, x1, viewIdx, :]

    # -------------------------
    # 进行加权
    # -------------------------
    sampled = (
        s00 * w00.unsqueeze(-1) +
        s01 * w01.unsqueeze(-1) +
        s10 * w10.unsqueeze(-1) +
        s11 * w11.unsqueeze(-1)
    )  # (B, C)

    return sampled
from typing import Any, Dict
def tocuda(obj: Any, device: str | torch.device = "cuda", non_blocking: bool = True) -> Any:
    """
    Recursively move all torch.Tensors inside a (nested) dict/list/tuple/set to CUDA (or a given device).

    Args:
        obj: Can be a Tensor, dict, list, tuple, set, or any other python object.
        device: Target device. Default "cuda".
        non_blocking: Passed to Tensor.to(...). Effective when source is pinned CPU memory.

    Returns:
        The same structure as obj, with all tensors moved to `device`.
    """
    if torch.is_tensor(obj):
        # Keep dtype; move device only
        return obj.to(device=device, non_blocking=non_blocking)

    if isinstance(obj, dict):
        return {k: tocuda(v, device=device, non_blocking=non_blocking) for k, v in obj.items()}

    if isinstance(obj, list):
        return [tocuda(v, device=device, non_blocking=non_blocking) for v in obj]

    if isinstance(obj, tuple):
        return tuple(tocuda(v, device=device, non_blocking=non_blocking) for v in obj)

    if isinstance(obj, set):
        return {tocuda(v, device=device, non_blocking=non_blocking) for v in obj}

    # Leave other objects unchanged (numbers, strings, None, etc.)
    return obj

def sample_impostor_uv_bilinear(coords, tex, device="cuda"):
    """
    手动 bilinear 从 impostor 采样：
    输入:
        coords: (B,3)  -> (u,v,viewIdx)
            u,v ∈ [-1,1]
            viewIdx: long

    输出:
        sampled: (B,C)
    """

    coords = coords.to(device)
    u = coords[:, 0]          # [-1,1]
    v = coords[:, 1]
    viewIdx = coords[:, 2].long()

    
    H, W, C = tex.shape

    # -------------------------
    # 将 [-1,1] 映射到 [0,W-1] / [0,H-1]
    # -------------------------
    uf = (u + 1) * 0.5 * (W - 1)     # float pixel space
    vf = (v + 1) * 0.5 * (H - 1)

    # 找到邻域像素
    x0 = uf.floor().long().clamp(0, W - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y0 = vf.floor().long().clamp(0, H - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    # bilinear 权重
    wx = uf - x0.float()   # (B,)
    wy = vf - y0.float()

    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy

    # -------------------------
    # 逐视角索引采样：VERY MEMORY EFFICIENT
    # -------------------------
    # tex: (H, W, V, C)
    s00 = tex[y0, x0, :]   # (B,C)
    s01 = tex[y1, x0, :]
    s10 = tex[y0, x1, :]
    s11 = tex[y1, x1, :]

    # -------------------------
    # 进行加权
    # -------------------------
    sampled = (
        s00 * w00.unsqueeze(-1) +
        s01 * w01.unsqueeze(-1) +
        s10 * w10.unsqueeze(-1) +
        s11 * w11.unsqueeze(-1)
    )  # (B, C)

    return sampled

import torch.nn as nn
class ViewEncoder(nn.Module):
    def __init__(self, in_ch=3, feat_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1,padding_mode="border"), nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2,padding_mode="border"), nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, feat_dim, 3, padding=1, stride=2,padding_mode="border"), nn.LeakyReLU(inplace=True)
        )
        self.feat_dim = feat_dim

    def forward(self, x):  # x: (V, C, H, W)
        return self.encoder(x)  # (V, F, H', W')

    
class RayEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(True),
            nn.Linear(64, out_dim)
        )

    def forward(self, raydir):  # (B,3)
        return self.net(raydir)

class MultiViewImpostorNetwork(nn.Module):
    def __init__(self, feat_dim=64, num_heads=4):
        super().__init__()
        self.view_encoder = ViewEncoder(in_ch=9,feat_dim=feat_dim)
        self.geo_encoder = ViewEncoder(in_ch=6,out_dim = feat_dim)
        self.ray_encoder = RayEncoder(in_dim=6,out_dim=feat_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, view_tex, uvi, raydir):
        """
        view_tex: (V, 3, H, W)
        uvi: (B, 3, 3)  → 每条光线对应 3 条 (u,v,view_id)
        raydir: (B, 3)
        """

        # ========= 1. 编码各视角图像 =========
        # Nts: (V, F, H', W')
        Nts = self.encoder(view_tex)

        V, F, Hp, Wp = Nts.shape
        B = uvi.shape[0]

        # ========= 2. 根据 (uv, view_id) 采样 3 个 feature =========
        feats = []
        for i in range(3):
            uv = uvi[:, i, :2]     # (B,2) [-1,1]
            vid = uvi[:, i, 2].long()  # (B,)
            feat_i = []

            for t in range(B):
                tex = Nts[vid[t]]        # (F, H', W')
                f = bilinear_sample(tex, uv[t:t+1])  # (1,F)
                feat_i.append(f)

            feat_i = torch.cat(feat_i, dim=0)  # (B,F)
            feats.append(feat_i)

        # stack: (B,3,F)
        feats = torch.stack(feats, dim=1)

        # ========= 3. 光线编码 =========
        ray_feat = self.ray_encoder(raydir).unsqueeze(1)  # (B,1,F)

        # ========= 4. Attention 融合 =========
        fused, _ = self.attn(query=ray_feat, key=feats, value=feats)
        #    fused: (B,1,F)

        return fused.squeeze(1), feats   # (B,F), (B,3,F)
    
from collections import defaultdict

def group_by_view_triplet_with_roughness_fast(sample, max_view=65535, max_rbin=255):
    """
    返回 buckets: dict[key_tuple] -> item(dict of tensors)
    key_tuple = (i0,i1,i2,rbin)

    关键优化：
    - 不做 per-pixel .item()
    - 不做 per-pixel Python dict append
    """
    roughness   = sample["roughness"].reshape(-1)     # (P,)
    mask        = sample["mask"].reshape(-1)          # (P,)

    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return {}

    # 取 triplet 和 rbin
    triplets = sample["sampleView"].long().reshape(-1,3)[valid_idx]    # (M,3)
    rbin = compute_roughness_bin(roughness)[valid_idx].long()  # (M,)

    # -------- pack key to int64: [i0 | i1 | i2 | rbin] -------
    # 这里用 16bit/16bit/16bit/8bit 举例；你可按实际范围调整
    if triplets.max() > max_view or rbin.max() > max_rbin:
        raise ValueError("max_view/max_rbin too small for your indices")

    key_id = (triplets[:, 0] << 40) | (triplets[:, 1] << 24) | (triplets[:, 2] << 8) | rbin
    #          16 bits          16 bits          16 bits        8 bits

    # -------- group by unique key id -------
    uniq, inv = torch.unique(key_id, return_inverse=True)  # uniq: (K,), inv:(M,)

    # 把同一桶的像素排到一起
    order = torch.argsort(inv)
    inv_sorted = inv[order]
    valid_sorted = valid_idx[order]  # 这是“全局像素 index”，你想要的

    # 每个桶的大小
    counts = torch.bincount(inv_sorted, minlength=uniq.numel())
    offsets = torch.cumsum(counts, dim=0)
    starts = offsets - counts

    # -------- flatten data once -------
    flat_data = {}
    for k, v in sample.items():
        if k in ["node"]:
            continue

        flat_data[k] = v.reshape(-1, v.shape[-1])
        
    # -------- build buckets (Python loop over K buckets, not M pixels) -------
    buckets = {}
    for bi in range(uniq.numel()):
        s = starts[bi].item()
        e = offsets[bi].item()
        idxs = valid_sorted[s:e]  # (count,)

        # decode key tuple (optional)
        kid = uniq[bi].item()
        i0 = (kid >> 40) & 0xFFFF
        i1 = (kid >> 24) & 0xFFFF
        i2 = (kid >> 8)  & 0xFFFF
        rb = kid & 0xFF
        key_tuple = (int(i0), int(i1), int(i2), int(rb))

        item = {}
        for k, arr in flat_data.items():
            item[k] = arr[idxs]

        item["mask"] = mask[idxs]
        buckets[key_tuple] = item

    return buckets

def group_by_view_triplet_with_roughness(sample):
    """
    按以下组合分桶：
        (view0, view1, view2, roughness_bin)

    sample["reflect_uvi"]: (B,3,3)
    sample["roughness"]: (H,W,1)
    sample["mask"]:      (H,W,1)  True/False
    """

    reflect_uvi = sample["reflect_uvi"]       # (B,3,3)
    roughness   = sample["roughness"].reshape(-1)  # (B,)
    mask        = sample["mask"].reshape(-1)       # (B,)
    B = reflect_uvi.shape[0]

    # ---------------------------
    # 过滤无效 pixel（mask=False）
    # ---------------------------
    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return {}

    # ---------------------------
    # 获取有效像素的视角 triplet (i0,i1,i2)
    # ---------------------------
    triplets = reflect_uvi[valid_idx, :, 2].long()     # shape = (M,3)

    # ---------------------------
    # 计算 roughness bin
    # ---------------------------
    rbin = compute_roughness_bin(roughness)[valid_idx]  # shape = (M,)
    # ---------------------------
    # 构造最终 key = (i0,i1,i2,rbin)
    # ---------------------------
    keys = [tuple([triplets[i,0].item(),
                   triplets[i,1].item(),
                   triplets[i,2].item(),
                   rbin[i].item()]) 
            for i in range(valid_idx.numel())]

    # ---------------------------
    # 建立 bucket → pixel index 列表
    # ---------------------------
    buckets_idx = defaultdict(list)
    for local_i, key in enumerate(keys):
        global_i = valid_idx[local_i].item()
        buckets_idx[key].append(global_i)

    # ---------------------------
    # 展平 tensors 以便批量索引
    # ---------------------------
    flat_data = {}
    for key, value in sample.items():
        if key in ["node"]:
            continue
        elif key in ["reflect_uvi", "first_uvi","reference_uvi","reference_uvi2"]:
            flat_data[key] = value.reshape(-1, value.shape[-2], value.shape[-1])
        else:
            flat_data[key] = value.reshape(-1, value.shape[-1])

    # ---------------------------
    # 构造最终 buckets
    # ---------------------------
    buckets = {}

    for triplet_rbin, pix_list in buckets_idx.items():

        idxs = torch.tensor(pix_list, dtype=torch.long, device=reflect_uvi.device)
        item = {}

        # 取所有 tensor key
        for key, arr in flat_data.items():
            item[key] = arr[idxs]

        # reflect_uvi / first_uvi 也要取
        item["reflect_uvi"] = sample["reflect_uvi"][idxs]
        item["reference_uvi"] = sample["reference_uvi"][idxs]
        item["reference_uvi2"] = sample["reference_uvi2"][idxs]
        if "first_uvi" in sample:
            item["first_uvi"] = sample["first_uvi"][idxs]

        # mask
        item["mask"] = mask[idxs]

        buckets[triplet_rbin] = item

    return buckets
import zstandard as zstd
import pickle


def compute_roughness_bin(roughness):
    """
    输入: roughness tensor (B, 1) or (B,)
    输出: rbin (B,)
    """
    r = roughness.reshape(-1)

    # 分段（向量化）
    rbin = torch.zeros_like(r, dtype=torch.long)

    rbin[r >= 0.05] = 1
    rbin[r >= 0.10] = 2
    rbin[r >= 0.20] = 3
    rbin[r >= 0.30] = 4

    return rbin

def save_bucket_chunk(SAVE_ROOT,scene_id, triplet, bucket_data, chunk_idx=0):
    """
    bucket_data: dict of tensors (N, C)
    保存前 N 条 pixel（所有 key 对应维度一致）
    """

    fn = f"bucket_{triplet[0]}_{triplet[1]}_{triplet[2]}_{triplet[3]}_part{chunk_idx}.pkl.zst"
    full_path = os.path.join(SAVE_ROOT, fn)

    # 转 numpy
    numpy_bucket = {k: v.cpu().numpy() for k, v in bucket_data.items()}

    enc = zstd.ZstdCompressor(level=3)
    packed = pickle.dumps(numpy_bucket)

    with open(full_path, "wb") as f:
        f.write(enc.compress(packed))

    print(f"[Saved] {full_path}  ({len(numpy_bucket[list(numpy_bucket.keys())[0]])} pixels)")
import time


def dataset_process_debug(path,impostor):
    MAX_BUCKET_SIZE = 512 * 512  # 65536
    SAVE_ROOT =path + "buckets_fix/"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    file_list = os.listdir(path)
    final_bucket = {}
    bucket_save_count = {}  
    for scene_str in file_list:
        scene_id = int(scene_str)
        folder = os.path.join(path, f"{scene_id}")
        sample = {}

        for key in  ["AccumulatePassoutput","albedo","depth","specular","normal","position","roughness","view","raypos","emission"]:
            exr_path = os.path.join(folder, f"{key}_{scene_id:05d}.exr")

            data = pyexr.read(exr_path)[..., : buffer_channel[key]]

            sample[key] = torch.Tensor(data).cuda()  # shape: (H,W,C)
        #pyexr.write("H:/Falcor/media/inv_rendering_scenes/object_level_config/Bunny/{}roughness.exr".format(scene_id),sample["roughness"].cpu().numpy())
        #sample["mask"] = (sample["position"].sum(dim=-1,keepdim=True)!=0) & (sample["roughness"] < 0.0001) & (sample["normal"][...,2:3]<0.1)
  
        json_path = os.path.join(folder, "node.json")
        with open(json_path, "r") as f:
            node = json.load(f)

        sample["node"] = node  # dict，不转 GPU
        train_data = sample
        W,H,_ =train_data["raypos"].shape
        
        rayPosW,rayDirW = train_data["raypos"].reshape(-1,3),-train_data["view"].reshape(-1,3)
        normal = train_data["normal"].reshape(-1,3)
        reflectPosW = train_data["position"].reshape(-1,3)
        reflectDirW = compute_reflection_direction(rayDirW,normal)
        
        print(torch.Tensor(train_data["node"]["Light"]["transform"]))
        rayPos_local, rayDir_local = transform_rays_world_to_local(
            rayPosW,
            rayDirW,
            torch.Tensor(train_data["node"]["Light"]["transform"])
        )
        refelctPosL, reflectDirL = transform_rays_world_to_local(
            reflectPosW,
            reflectDirW,
            torch.Tensor(train_data["node"]["Light"]["transform"])
        )
        
        first_uvi = chunk_process(
            impostor,
            rayDir_local,
            rayPos_local,
            W=W,
            H=H,
            chunk=512 * 512 // 8   # 可根据显存调整
        )
        reflect_uvi = chunk_process(
            impostor,
            reflectDirL,
            refelctPosL,
            W=W,
            H=H,
            chunk=512 * 512 // 8   # 可根据显存调整
        )
        sample["reflect_uvi"] = reflect_uvi
        sample["first_uvi"] = reflect_uvi
        sample["refelctPosL"] = refelctPosL
        sample["reflectDirL"] = reflectDirL
        # pyexr.write("H:/Falcor/media/inv_rendering_scenes/object_level_config/Bunny/{}mask.exr".format(scene_id),sample["mask"].reshape(H,W,1).cpu().numpy())
        # for i in range(3):
        #     reflect_emission = sample_impostor_uv_bilinear(impostor,reflect_uvi[:,i,:],tex_name="emission")
        #     pyexr.write("H:/Falcor/media/inv_rendering_scenes/object_level_config/Bunny/{}reflect_uvi{}.exr".format(scene_id,i),reflect_uvi[:,i,:].reshape(H,W,3).cpu().numpy())
        #     pyexr.write("H:/Falcor/media/inv_rendering_scenes/object_level_config/Bunny/{}debug_emission{}.exr".format(scene_id,i),reflect_emission[:,:].reshape(H,W,3).cpu().numpy())
        M = torch.tensor(train_data["node"]["Light"]["transform"], dtype=torch.float32).cuda()
        
        save_bucket_chunk(SAVE_ROOT,scene_id, triplet, chunk_data, bucket_save_count[triplet])
                
    return sample,scene_id



def process_single_scene(args):
    path, impostor, scene_str = args

    scene_id = int(scene_str)
    folder = os.path.join(path, f"{scene_id}")

    print(f"[PID {os.getpid()}] processing scene {scene_id}")

    sample = {}

    # 读取 EXR 到 GPU
    for key in ["AccumulatePassoutput","albedo","depth",
                "specular","normal","position","roughness",
                "view","raypos","emission"]:
        exr_path = os.path.join(folder, f"{key}_{scene_id:05d}.exr")
        data = pyexr.read(exr_path)[..., : buffer_channel[key]]
        sample[key] = torch.tensor(data, dtype=torch.float32).cuda()

    # mask
    sample["mask"] = (sample["position"].sum(dim=-1, keepdim=True) != 0) & \
                     (sample["roughness"] > 0.01)

    # node.json
    json_path = os.path.join(folder, "node.json")
    with open(json_path, "r") as f:
        node = json.load(f)
    sample["node"] = node

    W, H, _ = sample["raypos"].shape

    # 展平
    rayPosW    = sample["raypos"].reshape(-1, 3)
    rayDirW    = -sample["view"].reshape(-1, 3)
    normal     = sample["normal"].reshape(-1, 3)
    reflectPosW = sample["position"].reshape(-1, 3)
    reflectDirW = compute_reflection_direction(rayDirW, normal)

    M = torch.tensor(sample["node"]["Light"]["transform"], dtype=torch.float32).cuda()

    # 世界 → 光源局部
    rayPos_local, rayDir_local = transform_rays_world_to_local(
        rayPosW, rayDirW, M
    )
    reflectPosL, reflectDirL = transform_rays_world_to_local(
        reflectPosW, reflectDirW, M
    )

    # impostor 采样
    first_uvi = chunk_process(
        impostor,
        rayDir_local,
        rayPos_local,
        W=W,
        H=H,
        chunk=512 * 512 // 8
    )
    reflect_uvi = chunk_process(
        impostor,
        reflectDirL,
        reflectPosL,
        W=W,
        H=H,
        chunk=512 * 512 // 8
    )

    sample["first_uvi"]  = first_uvi
    sample["reflect_uvi"] = reflect_uvi

    # 分 bucket
    buckets = group_by_view_triplet_with_roughness(sample)

    # 注意：跨进程传输 GPU Tensor 会出问题，这里统一搬回 CPU
    cpu_buckets = {}
    for triplet, bucket_data in buckets.items():
        cpu_buckets[triplet] = {
            k: v.detach().cpu() for k, v in bucket_data.items()
        }

    return scene_id, cpu_buckets


import multiprocessing as mp

def dataset_process_mp(path, impostor, num_workers=8):
    MAX_BUCKET_SIZE = 256 * 256
    SAVE_ROOT = os.path.join(path, "buckets")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    file_list = os.listdir(path)
    # 只保留纯数字的子目录
    scene_list = [f for f in file_list if f.isdigit()]
    scene_list.sort(key=lambda x: int(x))

    final_bucket = {}
    bucket_save_count = {}

    # 准备参数
    args_list = [(path, impostor, scene_str) for scene_str in scene_list]

    with mp.Pool(processes=num_workers) as pool:
        # imap_unordered 可以边算边合并
        for scene_id, buckets in pool.imap_unordered(process_single_scene, args_list):
            print(f"[Main] merge scene {scene_id}")

            for triplet, bucket_data in buckets.items():

                if triplet not in final_bucket:
                    # 初始化
                    final_bucket[triplet] = {
                        k: v.clone() for k, v in bucket_data.items()
                    }
                    bucket_save_count[triplet] = 0
                else:
                    # 追加
                    for k in final_bucket[triplet]:
                        final_bucket[triplet][k] = torch.cat(
                            [final_bucket[triplet][k], bucket_data[k]],
                            dim=0
                        )

                # 检查是否超阈值，和你原来逻辑一样
                N = final_bucket[triplet]["reflect_uvi"].shape[0]
                while N > MAX_BUCKET_SIZE:
                    chunk_data = {
                        k: v[:MAX_BUCKET_SIZE] for k, v in final_bucket[triplet].items()
                    }
                    save_bucket_chunk(
                        SAVE_ROOT,
                        scene_id,          # 和你原来一样，用当前 scene_id 作为文件名的一部分
                        triplet,
                        chunk_data,
                        bucket_save_count[triplet]
                    )
                    bucket_save_count[triplet] += 1

                    # 剩余的留在内存里
                    final_bucket[triplet] = {
                        k: v[MAX_BUCKET_SIZE:] for k, v in final_bucket[triplet].items()
                    }
                    N = final_bucket[triplet]["reflect_uvi"].shape[0]

    # 所有 scene 处理完后，final_bucket 里可能还有没满块的，按需再保存一次
    for triplet, data in final_bucket.items():
        N = data["reflect_uvi"].shape[0]
        if N > 0:
            save_bucket_chunk(
                SAVE_ROOT,
                -1,   # 或者用特殊 scene_id 标记“剩余”
                triplet,
                data,
                bucket_save_count[triplet]
            )

    print("[Main] dataset_process_mp done.")


def compute_uvi(train_data,impostor,W,H):
    rayPosW,rayDirW = train_data["raypos"].reshape(-1,3),-train_data["view"].reshape(-1,3)
    normal = train_data["normal"].reshape(-1,3)
    reflectPosW = train_data["position"].reshape(-1,3)
    reflectDirW = compute_reflection_direction(rayDirW,normal)
    
    print(torch.Tensor(train_data["node"]["Light"]["transform"]))
    rayPos_local, rayDir_local = transform_rays_world_to_local(
        rayPosW,
        rayDirW,
        torch.Tensor(train_data["node"]["Light"]["transform"])
    )
    refelctPosL, reflectDirL = transform_rays_world_to_local(
        reflectPosW,
        reflectDirW,
        torch.Tensor(train_data["node"]["Light"]["transform"])
    )
    
    first_uvi = chunk_process(
        impostor,
        rayDir_local,
        rayPos_local,
        W=W,
        H=H,
        chunk=512 * 512 // 8   # 可根据显存调整
    )
    reflect_uvi = chunk_process(
        impostor,
        reflectDirL,
        refelctPosL,
        W=W,
        H=H,
        chunk=512 * 512 // 8   # 可根据显存调整
        )
  


def dataset_process_new(path,impostor,test=False):
    key_list = ["color","position","albedo","specular","normal","roughness","depth","emission","AccumulatePassoutput","view","raypos","mind","reflect","idepth0","idepth1","idepth2","idirection0","idirection1","idirection2"]
    roughness = torch.round(torch.arange(0.0, 1.0 + 1e-9, 0.01) * 100) / 100
    theta95_deg = ggx_theta99_from_roughness(roughness, alpha_is_roughness_sq=True, degrees=True).cuda()
    MAX_BUCKET_SIZE = 256 * 256  # 65536
    SAVE_ROOT =path + "zst/"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    file_list = os.listdir(path)
    final_bucket = {}
    bucket_save_count = {}  
    cnt = 0
    for scene_str in file_list:
        if scene_str == "zst":
            continue
        scene_id = int(scene_str)
        folder = os.path.join(path, f"{scene_id}")
        sample = {}

        for key in  key_list:
            exr_path = os.path.join(folder, f"{key}_{scene_id:05d}.exr")

            data = pyexr.read(exr_path)[..., : buffer_channel[key]]

            sample[key] = torch.Tensor(data).cuda()  # shape: (H,W,C)
        #pyexr.write("H:/Falcor/media/inv_rendering_scenes/object_level_config/Bunny/{}roughness.exr".format(scene_id),sample["roughness"].cpu().numpy())
        sample["mask"] = (sample["position"].sum(dim=-1,keepdim=True)!=0) & (sample["roughness"] < 0.0001) & (sample["normal"][...,2:3]<0.1) & (sample["mind"] != 1000)
        #sample["mask"] = (sample["position"].sum(dim=-1,keepdim=True)!=0) & (sample["roughness"] > 0.0001) & (sample["normal"][...,2:3]<0.1)
        if sample["mask"].sum() < 1:
            print("continue ",scene_id)
            continue
        json_path = os.path.join(folder, "node.json")
        with open(json_path, "r") as f:
            node = json.load(f)
        half_angle_deg = theta95_deg[(sample["roughness"].reshape(-1)*100).long()].reshape(-1)
        sample["node"] = node  # dict，不转 GPU

        W,H,_ =sample["raypos"].shape
        
        rayPosW,rayDirW = sample["raypos"].reshape(-1,3),-sample["view"].reshape(-1,3)
        normal = sample["normal"].reshape(-1,3)
        reflectPosW = sample["position"].reshape(-1,3)
        reflectDirW = compute_reflection_direction(rayDirW,normal)
        
        print(torch.Tensor(sample["node"]["Light"]["transform"]))
        rayPos_local, rayDir_local = transform_rays_world_to_local(
            rayPosW,
            rayDirW,
            torch.Tensor(sample["node"]["Light"]["transform"])
        )
        refelctPosL, reflectDirL = transform_rays_world_to_local(
            reflectPosW,
            reflectDirW,
            torch.Tensor(sample["node"]["Light"]["transform"])
        )
        
        sample["refelctPosL"] = refelctPosL
        sample["reflectDirL"] = reflectDirL
        reference_pos = refelctPosL + reflectDirL * sample["mind"].reshape(-1,1) / 0.8
        cone_radius = cone_radius_deg(sample["mind"].reshape(-1,1) / 0.8,half_angle_deg.reshape(-1,1))
        reference_pos2 = refelctPosL + reflectDirL * (sample["mind"].reshape(-1,1) / 0.8 +  cone_radius)
        sample["referencePosL"] = (refelctPosL + reflectDirL * sample["mind"].reshape(-1,1) / 0.8).reshape(512,512,3)
        sample["referencePosL2"] = reference_pos2.reshape(512,512,3)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first_uvi,_ = chunk_process(impostor, rayDir_local, rayPos_local, W=W, H=H, chunk=512*512//8)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[TIMER] chunk_process(first_uvi): {(t1 - t0)*1000:.3f} ms")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
      
  
        reflect_uvi,_ = chunk_process(impostor, reflectDirL, refelctPosL, W=W, H=H, chunk=512*512//8 )
        sample["reflect_uvi"] = reflect_uvi.reshape(512,512,3,3)
        sample["first_uvi"] = first_uvi.reshape(512,512,3,3)
        
        reference_uvi,_ =  chunk_process(impostor, reflectDirL, refelctPosL, W=W, H=H, chunk=512*512//8 ,reference_pos=reference_pos)
        reference_uvi2,_ =  chunk_process(impostor, reflectDirL, refelctPosL, W=W, H=H, chunk=512*512//8 ,reference_pos=reference_pos2)
        sample["reference_uvi"] = reference_uvi.reshape(512,512,3,3)
        sample["reference_uvi2"] = reference_uvi2.reshape(512,512,3,3)
        N = rayDir_local.shape[0]
        out_uvi = []

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[TIMER] chunk_process(reflect_uvi): {(t1 - t0)*1000:.3f} ms")
        pos = sample["refelctPosL"]
        dir = sample["reflectDirL"]
        rr = sample["roughness"]

        halfAngle = theta95_deg[(rr * 100).long()]
        halfRadius = halfAngle * np.pi / 180.0
        mask = sphere_covers_cone(pos,dir,halfRadius,torch.Tensor([impostor.radius]).cuda())
        sample["mask"] = sample["mask"] & mask.reshape(512,512,1)
        if test:
            ensure_dir(os.path.join(path,"test"))
            fn = f"{cnt}.pkl.zst"
            full_path = os.path.join(path,"test", fn)

            # 转 numpy

            enc = zstd.ZstdCompressor(level=3)
            packed = pickle.dumps(sample)

            with open(full_path, "wb") as f:
                f.write(enc.compress(packed))
            cnt = cnt+ 1
            continue
        M = torch.tensor(sample["node"]["Light"]["transform"], dtype=torch.float32).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        buckets = group_by_view_triplet_with_roughness_fast(sample)
        #buckets2 = group_by_view_triplet_with_roughness(sample)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[TIMER] group_by_view_triplet_with_roughness(reflect_uvi): {(t1 - t0)*1000:.3f} ms")
        #continue
        for triplet, bucket_data in buckets.items():

            # 初始化 final bucket
            if triplet not in final_bucket:
                final_bucket[triplet] = {k: v.clone() for k, v in bucket_data.items()}
                bucket_save_count[triplet] = 0
            else:
                # 追加（拼接）
                for key in final_bucket[triplet]:
                    final_bucket[triplet][key] = torch.cat([
                        final_bucket[triplet][key],
                        bucket_data[key]
                    ], dim=0)

            # 检查长度是否超过阈值
            N = final_bucket[triplet]["reflect_uvi"].shape[0]

            while N > MAX_BUCKET_SIZE:

                # 截取前 MAX_BUCKET_SIZE 条保存
                chunk_data = {k: v[:MAX_BUCKET_SIZE] for k, v in final_bucket[triplet].items()}
                save_bucket_chunk(SAVE_ROOT,scene_id, triplet, chunk_data, bucket_save_count[triplet])
                bucket_save_count[triplet] += 1

                # 删除已保存的部分，保留剩余
                final_bucket[triplet] = {k: v[MAX_BUCKET_SIZE:] for k, v in final_bucket[triplet].items()}
                N = final_bucket[triplet]["reflect_uvi"].shape[0]

    return sample,scene_id