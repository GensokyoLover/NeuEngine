import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from futils import  *
import os
import json
import cv2
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

def build_lookup_texture(verts, faces, resolution=256):
    # 计算每个面法线
    face_normals = np.cross(
        verts[faces[:,1]] - verts[faces[:,0]],
        verts[faces[:,2]] - verts[faces[:,0]]
    )
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]

    lookup = np.zeros((resolution, resolution), dtype=np.int32)

    # 为每个 texel 生成方向
    for y in range(resolution):
        for x in range(resolution):
            u = (x + 0.5) / resolution * 2 - 1
            v = (y + 0.5) / resolution * 2 - 1
            dir = np.array([u, v, 1 - abs(u) - abs(v)])
            if dir[2] < 0:
                dir[:2] = (1 - np.abs(dir[:2][::-1])) * np.sign(dir[:2])
            dir /= np.linalg.norm(dir)

            # 找最近的面法线（可替换成KDTree以加速）
            dots = face_normals @ dir
            idx = np.argmax(dots)
            lookup[y, x] = idx

    return lookup

def build_lookup_texture_speedup(verts, faces, resolution=256):
    # === 计算每个面法线 ===
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

    # === 生成所有方向向量 (octahedral map sampling) ===
    # u,v ∈ [-1,1]
    grid = np.linspace(0, resolution - 1, resolution)
    u, v = np.meshgrid(grid, grid, indexing="xy")
    u = (u + 0.5) / resolution * 2 - 1
    v = (v + 0.5) / resolution * 2 - 1

    # dir_z = 1 - |u| - |v|
    dir_z = 1 - np.abs(u) - np.abs(v)
    dir = np.stack([u, v, dir_z], axis=-1)

    # 处理下半球：如果 z<0，则折叠边缘
    mask = dir[..., 2] < 0
    dir_xy = dir[..., :2]
    dir_xy_swapped = dir_xy[..., ::-1]
    dir_xy_new = (1 - np.abs(dir_xy_swapped)) * np.sign(dir_xy)
    dir[..., :2][mask] = dir_xy_new[mask]

    # 归一化所有方向
    dir /= np.linalg.norm(dir, axis=-1, keepdims=True)

    # === 展平后批量计算 dot(face_normals, dir) ===
    dirs_flat = dir.reshape(-1, 3)
    # (H*W, F) = (H*W,3) @ (3,F)
    dots = dirs_flat @ face_normals.T
    lookup_flat = np.argmax(dots, axis=1)
    lookup = lookup_flat.reshape(resolution, resolution)

    return lookup

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

def generate_impostor_by_falcor(testbed,scene_path,output_path,object_data_dict,sellect_list):
    testbed.load_scene(scene_path)
    scene = testbed.scene
    a = f3_to_numpy(scene.bounds.min_point)
    b = f3_to_numpy(scene.bounds.max_point)
    centor,object_radius,o_radius,p_radius = sampling_radii_from_aabb(a,b)
    final_resolution = 512
    testbed.scene.camera.focalLength = 0
    testbed.scene.camera.nearPlane = 0.001
    testbed.scene.add_impostor()
    base = 1
    basic_info = {}
    basic_info["radius"] = o_radius
    basic_info["centor"] = centor
    
    for subdiv_level in range(4):
        testbed.resize_frame_buffer(512//base,512//base)
        basic_info["inv_dim"] = 1/(512//base)

        level_output_path = output_path + "level{}/".format(subdiv_level)
        if not os.path.exists(level_output_path):
            os.makedirs(level_output_path)

        verts, faces = geodesic_impostor_mesh(subdiv_level)
        faces_list = faces.tolist()  
        #lookup_table = build_lookup_texture(verts,faces,resolution=128)
        lookup_table = build_lookup_texture_speedup(verts,faces,resolution=2048)
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
        for single_pos in camera_positions:
            testbed.scene.camera.position = single_pos
            testbed.scene.camera.target = normalize(centor - single_pos) * o_radius * 2 + single_pos
            up = unity_style_up(normalize(centor - single_pos))
            r,u,f = compute_camera_basis(single_pos,normalize(centor - single_pos) * o_radius * 2 + single_pos,up)
            r_list.append(r)
            u_list.append(u)
            f_list.append(f)
            p_list.append(single_pos)
            testbed.scene.camera.up = up
            testbed.run()

            for name in sellect_list:
                index = object_data_dict[name]
                tex2 = testbed.capture_output(
                    level_output_path + '{}_{}.exr'.format(name, cnt), index)
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
        
        position_path = os.path.join(level_output_path, "position.json")
        with open(position_path, "w") as f:
            json.dump([p_.tolist() for p_ in p_list], f, indent=4)
        print(f"✅ Saved right/up/forward/position JSONs to {level_output_path}")
            