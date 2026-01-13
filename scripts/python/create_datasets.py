import sys
import os
import platform

os_name = platform.system()
falcor_python_dir = None
if os_name == "Windows":
    falcor_dir = r"H:/Falcor\\build\\windows-vs2022\\bin\\Debug"  # Double \\ for raw string
    os.environ["PATH"] = falcor_dir + ";" + os.environ.get("PATH", "")
    falcor_python_dir = falcor_dir + r"\python"
else:  # Linux
    falcor_dir = r"/seaweedfs_tmp/training/wangjiu/new/NeuEngine/build/linux-clang/bin/Debug"
    os.environ["PATH"] = falcor_dir + ":" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = falcor_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    falcor_python_dir = falcor_dir + "/python"

# Prepend for higher priority (better than append)
sys.path.insert(0, falcor_python_dir)

import falcor
import numpy as np
from impostor import *
from futils import f3_to_numpy

def setup_renderpass(testbed):
    render_graph = testbed.create_render_graph("PathTracer")
    render_graph.create_pass("PathTracer", "PathTracer", {'samplesPerPixel': 1})
    render_graph.create_pass("VBufferRT", "VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    render_graph.create_pass("AccumulatePass", "AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    render_graph.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    render_graph.add_edge("PathTracer.color", "AccumulatePass.input")
    render_graph.mark_output("AccumulatePass.output")
    testbed.render_graph = render_graph

def render_graph_MinimalPathTracer_Old(testbed):
    g = testbed.create_render_graph("MinimalPathTracer")
    g.create_pass("AccumulatePass","AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.create_pass("ToneMapper","ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.create_pass("MinimalPathTracer","MinimalPathTracer", {'maxBounces': 1})
    g.create_pass("VBufferRT","VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "MinimalPathTracer.viewW")
    g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")
    g.markOutput("MinimalPathTracer.color")
    
    g.markOutput("MinimalPathTracer.position")
    g.markOutput("MinimalPathTracer.albedo")
    g.markOutput("MinimalPathTracer.specular")
    g.markOutput("MinimalPathTracer.normal")
    g.markOutput("MinimalPathTracer.roughness")
    g.markOutput("MinimalPathTracer.depth")
    g.markOutput("MinimalPathTracer.emission")
    g.markOutput("AccumulatePass.output")
    g.markOutput("MinimalPathTracer.view")
    g.markOutput("MinimalPathTracer.raypos")
    g.markOutput("MinimalPathTracer.mind")
    testbed.render_graph = g


def render_graph_ImpostorTracer(testbed):
    g = testbed.create_render_graph("ImpostorTracer")
    g.create_pass("AccumulatePass","AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.create_pass("ToneMapper","ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.create_pass("ImpostorTracer","ImpostorTracer", {'maxBounces': 3})
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("ImpostorTracer.color", "AccumulatePass.input")
    g.markOutput("ImpostorTracer.color")
    g.markOutput("ImpostorTracer.position")
    g.markOutput("ImpostorTracer.albedo")
    g.markOutput("ImpostorTracer.specular")
    g.markOutput("ImpostorTracer.normal")
    g.markOutput("ImpostorTracer.roughness")
    g.markOutput("ImpostorTracer.depth")
    g.markOutput("AccumulatePass.output")
    testbed.render_graph = g
def render_graph_MinimalPathTracer(testbed,impostor_cnt):
    g = testbed.create_render_graph("MinimalPathTracer")
    g.create_pass("AccumulatePass", "AccumulatePass",{'enabled': True, 'precisionMode': 'Single'})
    g.create_pass("AccumulatePass2", "AccumulatePass",{'enabled': True, 'precisionMode': 'Single'})
    
    g.create_pass("ToneMapper", "ToneMapper",{'autoExposure': False, 'exposureCompensation': 0.0})
    g.create_pass("MinimalPathTracer","MinimalPathTracer", {'maxBounces': 6})
    
    g.create_pass("VBufferRT","VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16,"useTraceRayInline":False})
    #g.create_pass("AccumulatePass2", "AccumulatePass",{'enabled': True, 'precisionMode': 'Single'})
    #g.create_pass("MinimalPathTracer2", "MinimalPathTracer",{'maxBounces': 0})
    g.create_pass("VBufferRT2","VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16,"useTraceRayInline":True})
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer")
    
    g.addEdge("VBufferRT.viewW", "MinimalPathTracer.viewW")
    g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")
    g.addEdge("MinimalPathTracer.type", "AccumulatePass2.input")
    #g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer2.vbuffer")
    # g.addEdge("VBufferRT.viewW", "MinimalPathTracer2.viewW")
    g.addEdge("MinimalPathTracer.position", "VBufferRT2.prePosition")
    g.addEdge("MinimalPathTracer.reflect", "VBufferRT2.preDirection")
    g.addEdge("MinimalPathTracer.roughness", "VBufferRT2.preRoughness")
    
    g.markOutput("MinimalPathTracer.color")
    g.markOutput("MinimalPathTracer.position")
    g.markOutput("MinimalPathTracer.albedo")
    g.markOutput("MinimalPathTracer.specular")
    g.markOutput("MinimalPathTracer.normal")
    g.markOutput("MinimalPathTracer.roughness")
    g.markOutput("MinimalPathTracer.depth")
    g.markOutput("MinimalPathTracer.emission")
    g.markOutput("AccumulatePass.output")
    g.markOutput("MinimalPathTracer.view")
    g.markOutput("MinimalPathTracer.raypos")
    g.markOutput("MinimalPathTracer.mind")
    g.markOutput("MinimalPathTracer.reflect")
    g.markOutput("AccumulatePass2.output")
    for i in range(impostor_cnt):
        g.markOutput("VBufferRT2.uv0_{}".format(i))
        g.markOutput("VBufferRT2.uv1_{}".format(i))
        g.markOutput("VBufferRT2.uv2_{}".format(i))
        g.markOutput("VBufferRT2.direction0_{}".format(i))
        g.markOutput("VBufferRT2.direction1_{}".format(i))
        g.markOutput("VBufferRT2.direction2_{}".format(i))
        g.markOutput("VBufferRT2.depth0_{}".format(i))
        g.markOutput("VBufferRT2.depth1_{}".format(i))
        g.markOutput("VBufferRT2.depth2_{}".format(i))

    testbed.render_graph = g
def render_graph_MinimalPathTracer_Debug(testbed):
    g = testbed.create_render_graph("MinimalPathTracer")
    VBufferRT = g.create_pass("VBufferRT", "VBufferRT",{'samplePattern': 'Stratified', 'sampleCount': 1,"useTraceRayInline":True,})
    #g.addPass(VBufferRT, "VBufferRT")

    g.markOutput("VBufferRT.depth0")
    g.markOutput("VBufferRT.depth1")
    g.markOutput("VBufferRT.depth2")
    g.markOutput("VBufferRT.direction0")
    g.markOutput("VBufferRT.direction1")
    g.markOutput("VBufferRT.direction2")
    testbed.render_graph = g


import numpy as np

def compute_obj_bounding_box(obj_path):
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # 顶点行
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
    
    if not vertices:
        raise ValueError("没有找到顶点数据")

    vertices = np.array(vertices)
    min_xyz = vertices.min(axis=0)
    max_xyz = vertices.max(axis=0)
    
    return min_xyz, max_xyz

def get_bounding_box(path):
    min_bound, max_bound = compute_obj_bounding_box(path)

    print("Bounding Box:")
    print("Min:", min_bound)
    print("Max:", max_bound)
import pyexr
import pickle
import zstandard as zstd

def save_compressed_pickle(data, file_path):
    pickled_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    cctx = zstd.ZstdCompressor()

    # 压缩数据
    compressed_data = cctx.compress(pickled_data)
    with open(file_path, 'wb') as f:
        f.write(compressed_data)


object_data_list = ["color","position","albedo","specular","normal","roughness","depth","emission","AccumulatePassoutput","view","raypos","mind","reflect","AccumulatePassoutput2"]
for i in range(5):
    object_data_list.append("uv0_{}".format(i))
    object_data_list.append("uv1_{}".format(i))
    object_data_list.append("uv2_{}".format(i))
    object_data_list.append("direction0_{}".format(i))
    object_data_list.append("direction1_{}".format(i))
    object_data_list.append("direction2_{}".format(i))
    object_data_list.append("depth0_{}".format(i))
    object_data_list.append("depth1_{}".format(i))
    object_data_list.append("depth2_{}".format(i))
object_key_dict = {name: i for i, name in enumerate(object_data_list)}
sellect_list = ["albedo","specular","normal","position","view","AccumulatePassoutput","roughness","raypos","depth","emission","mind","reflect","AccumulatePassoutput2"]
for i in range(5):
    sellect_list.append("uv0_{}".format(i))
    sellect_list.append("uv1_{}".format(i))
    sellect_list.append("uv2_{}".format(i))
    sellect_list.append("direction0_{}".format(i))
    sellect_list.append("direction1_{}".format(i))
    sellect_list.append("direction2_{}".format(i))
    sellect_list.append("depth0_{}".format(i))
    sellect_list.append("depth1_{}".format(i))
    sellect_list.append("depth2_{}".format(i))
def pack_object_data(path,camera_resolution,direction_resolution):
    data = {}
    for name in object_data_list:
        data[name] = pyexr.read(path + name + ".exr").reshape(camera_resolution,direction_resolution,camera_resolution,direction_resolution,-1).transpose(2,0,1,3,4)
    save_compressed_pickle(data,path + "light.pkl.zst")


import os
import pyexr
import numpy as np
import random


def _uniform_step01(a: float, b: float) -> float:
    """
    在 [a, b] 上以 0.01 为最小步长均匀采样（包含端点，若端点不在 0.01 网格上会自动对齐）。
    """
    lo = math.ceil(a * 100)
    hi = math.floor(b * 100)
    if lo > hi:
        raise ValueError(f"Invalid range after 0.01-quantization: [{a}, {b}]")
    return random.randint(lo, hi) / 100.0


def sample_roughness() -> float:

    return _uniform_step01(0.01, 0.15)

import multiprocessing as mp

def save_pyscene_with_worker_id(src_path, worker_id, appended_commands):
    """
    src_path: 原始 .pyscene 路径，例如 "scene/xxx.pyscene"
    worker_id: int 或 str
    appended_commands: 要追加的字符串
    """

    base, ext = os.path.splitext(src_path)
    assert ext == ".pyscene"

    dst_path = f"{base}_worker{worker_id}{ext}"

    # 1. 读取源文件
    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. 拼接新内容
    new_content = (
        content.rstrip() + "\n\n"
        f"# ==== AUTO GENERATED FOR WORKER {worker_id} ====\n"
        + appended_commands.strip() + "\n"
    )

    # 3. 写入新文件
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return dst_path
import textwrap

def make_light_cmd(texture_name):
    IMPOSTOR_TEMPLATE = textwrap.dedent("""\
lightMesh = TriangleMesh()

normal = float3(0, 1, 0)

# 四个角
lightMesh.addVertex(float3(-0.7071, 0,  0.7071), normal, float2(0, 0))
lightMesh.addVertex(float3( 0.7071, 0,  0.7071), normal, float2(1, 0))
lightMesh.addVertex(float3( 0.7071, 0, -0.7071), normal, float2(1, 1))
lightMesh.addVertex(float3(-0.7071, 0, -0.7071), normal, float2(0, 1))

# 两个三角形
lightMesh.addTriangle(0, 1, 2)
lightMesh.addTriangle(0, 2, 3)


light = StandardMaterial('Light')
light.baseColor = float4(0)
light.emissiveColor = float3(0, 0, 0)
light.emissiveFactor = 1
light.roughness = 1
light.metallic = 1
sceneBuilder.addMeshInstance(
    sceneBuilder.addNode('Light', Transform(scaling=1.0, translation=float3(0, 0.0, 0.0), rotationEulerDeg=float3(0, 0, 0))),
    sceneBuilder.addTriangleMesh(lightMesh, light)
)
sceneBuilder.addTextureSlot("Light",r"H:\Falcor\emissive_crop\\{texture_name}","Emissive")
""")
    cmd = IMPOSTOR_TEMPLATE.format(
        texture_name = texture_name
    )
    return cmd
def make_cmd(name,mesh_path,impostor_path,emissive,translation,rotate,scale,roughness):
    IMPOSTOR_TEMPLATE = textwrap.dedent("""\
{obj_name} = StandardMaterial('{obj_name}')
{obj_name}.baseColor = float4(1.0, 1.0, 1.0, 1.0)
{obj_name}.emissiveColor = float3({emi_r}, {emi_g}, {emi_b})
{obj_name}.emissiveFactor = 1.0
{obj_name}.metallic = 1.0
{obj_name}.roughness = {roughness}

flags = TriangleMeshImportFlags.GenSmoothNormals | TriangleMeshImportFlags.JoinIdenticalVertices
{obj_name}Mesh = TriangleMesh.createFromFile(
    r'{mesh_path}',
    flags
)

sceneBuilder.addMeshInstance(
    sceneBuilder.addNode(
        '{obj_name}',
        Transform(
            translation=float3({tx}, {ty}, {tz}),
            rotationEulerDeg=float3({rx}, {ry}, {rz}),
            scaling={scale}
        )
    ),
    sceneBuilder.addTriangleMesh({obj_name}Mesh, {obj_name}, True)
)

sceneBuilder.addImpostor(
    r'{impostor_path}',
    r'{obj_name}'
)
""")
    cmd = IMPOSTOR_TEMPLATE.format(
        obj_name=name,
        emi_r=emissive[0], emi_g=emissive[1], emi_b=emissive[2],

        mesh_path=mesh_path,
        impostor_path=impostor_path,

        tx=translation[0], ty=translation[1], tz=translation[2],
        rx=rotate[0], ry=rotate[1], rz=rotate[2],
        scale=scale,
        roughness = roughness
    )
    return cmd
import trimesh

def sample_edge_biased(min_v, max_v, power=3.0):
    """
    power > 1  → 越大越靠近边缘
    power = 1  → 均匀分布
    """
    u = random.random()          # [0,1]
    sign = -1 if random.random() < 0.5 else 1
    x = sign * (u ** (1.0 / power))  # 偏向 ±1
    return 0.5 * (max_v + min_v) + 0.5 * (max_v - min_v) * x


# 辅助函数：检测两个 AABB 是否相交
def check_aabb_intersection(min_a, max_a, min_b, max_b):
    # 如果在任何一个轴上不重叠，则整体不相交
    if (max_a[0] < min_b[0] or min_a[0] > max_b[0]): return False
    if (max_a[1] < min_b[1] or min_a[1] > max_b[1]): return False
    if (max_a[2] < min_b[2] or min_a[2] > max_b[2]): return False
    return True

# 辅助函数：根据变换计算新的世界空间 AABB
# 比 transform 整个 mesh 快得多，只变换 8 个顶点
def get_transformed_aabb(mesh_min, mesh_max, scale_val, rotation_deg_y, translation):
    # 1. 构建局部空间的 8 个角点
    corners = np.array([
        [mesh_min[0], mesh_min[1], mesh_min[2]],
        [mesh_min[0], mesh_min[1], mesh_max[2]],
        [mesh_min[0], mesh_max[1], mesh_min[2]],
        [mesh_min[0], mesh_max[1], mesh_max[2]],
        [mesh_max[0], mesh_min[1], mesh_min[2]],
        [mesh_max[0], mesh_min[1], mesh_max[2]],
        [mesh_max[0], mesh_max[1], mesh_min[2]],
        [mesh_max[0], mesh_max[1], mesh_max[2]],
    ])

    # 2. 应用缩放 (Scale)
    corners = corners * scale_val

    # 3. 应用旋转 (Rotation - 绕Y轴)
    # 将角度转弧度
    rad = np.radians(rotation_deg_y)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    # Y轴旋转矩阵
    R = np.array([
        [cos_a,  0, sin_a],
        [0,      1, 0],
        [-sin_a, 0, cos_a]
    ])
    corners = np.dot(corners, R.T) # 矩阵乘法

    # 4. 应用平移 (Translation)
    corners = corners + np.array(translation)

    # 5. 获取新的 AABB
    new_min = np.min(corners, axis=0)
    new_max = np.max(corners, axis=0)
    return new_min, new_max

# --- 主逻辑开始 ---

# 用于存储已放置物体的 AABB [(min, max), (min, max), ...]

def generate_scene_data(impostor_list, model_path_fmt, max_attempts=100):
    """
    生成不重叠的物体变换和 Emissive 数据
    
    Returns:
        list[dict]: 包含每个物体 name, translate, scale, rotate, emissive 的字典列表
    """
    scene_data = []
    placed_aabbs = [] # 记录已放置物体的 AABB [(min, max), ...]

    for name in impostor_list:
        mp = model_path_fmt.format(name)
        
        # 加载 Mesh 获取原始包围盒 (这是碰撞检测必须的)
        try:
            mesh = trimesh.load(mp, force='mesh')
        except Exception as e:
            print(f"Error loading mesh {mp}: {e}")
            continue

        raw_aabb_min, raw_aabb_max = mesh.bounds
        success = False

        # 尝试生成无碰撞的变换
        for attempt in range(max_attempts):
            # 1. 随机参数
            scale = random.uniform(0.1, 0.4)
            emissive = np.random.uniform(0.2, 1, [3]) # 生成随机自发光颜色
            
            # 注意: 这里的 sample_edge_biased 需确保在上下文可见
            translate_vec = [
                sample_edge_biased(-0.707 + raw_aabb_max[0] * scale, 0.707 - raw_aabb_max[0] * scale, power=3.0),
                sample_edge_biased(-0.707 + raw_aabb_max[1] * scale, -0.707 + raw_aabb_max[1] * scale, power=3.0),
                sample_edge_biased(-0.707 + raw_aabb_max[2] * scale, 0.707 - raw_aabb_max[2] * scale, power=3.0)
            ]
            
            rot_y = random.uniform(0, 180)
            rotate_vec = [0, rot_y, 0]

            # 2. 计算新 AABB 并检测碰撞
            current_min, current_max = get_transformed_aabb(
                raw_aabb_min, raw_aabb_max, scale, rot_y, translate_vec
            )

            collision = False
            for (existing_min, existing_max) in placed_aabbs:
                if check_aabb_intersection(current_min, current_max, existing_min, existing_max):
                    collision = True
                    break
            
            # 3. 无碰撞则保存数据
            if not collision:
                placed_aabbs.append((current_min, current_max))
                
                # 构建纯数据字典
                obj_data = {
                    "name": name,
                    "translate": np.array(translate_vec),
                    "scale": np.array([scale, scale, scale]),
                    "rotate": np.array(rotate_vec),
                    "emissive": emissive
                }
                scene_data.append(obj_data)
                
                success = True
                break 

        if not success:
            print(f"Warning: Skipped {name} (overlap detected).")

    return scene_data

def worker_process(worker_id, resolution, scene_name,
                   object_data_dict, select_list,
                   task_queue: mp.Queue,result_queue: mp.Queue,impostor_list:list,emissive_list:list):
    """
    Worker 常驻进程：保持一个 Testbed 实例
    """
    print(f"[Worker {worker_id}] Initializing testbed...")
    
    os_name = platform.system()
    if os_name == "Windows":
        scene_path = r"H:\Falcor\scenes\scene/{}.pyscene".format(scene_name)
        model_path =r"H:\Falcor/model/{}.obj"
        impostor_path =r"H:\Falcor/datasets/impostor/{}/level1/"
        config_path =r"H:\Falcor\configs/{}.json".format(scene_name)
        emissive_path = r"H:\Falcor\emissive_crop\\{}"
    with open(config_path,'r') as f:
        configs = json.load(f)
    # 只有 worker 进程初始化 Falcor
    device = falcor.Device(
        type=falcor.DeviceType.Vulkan,
        gpu=0,
        enable_debug_layer=False
    )
    
    testbed = falcor.Testbed(
        width=resolution[0],
        height=resolution[1],
        create_window=False,
        device=device,
        spp=500
    )
    render_graph_MinimalPathTracer(testbed,len(impostor_list) + 5)
    cmd_list = []
    object_transform = []
    scene_data = generate_scene_data(impostor_list,model_path,100)
    i =0
    for impostor in impostor_list:
        name = impostor
        mp= model_path.format(name)
        mesh = trimesh.load(mp, force='mesh')
        aabb_min, aabb_max = mesh.bounds
        ip= impostor_path.format(name)
        scale = random.uniform(0.1,0.3)
        with open(ip + "material.json", 'r') as f:
            material_data = json.load(f)
            emissive = material_data.get("emissive", [0.0, 0.0, 0.0])
            roughness = material_data.get("roughness", 0.5)
        
        tf = {}

    
        rotate = [0,random.uniform(0,180),0]
        ##tf["translate"] = np.array(translate)
        tf["translate"] = np.array(scene_data[i]["translate"])
        tf["scale"] = np.array(scene_data[i]["scale"])
        tf["rotate"] = np.array(scene_data[i]["rotate"])
        object_transform.append(tf)
        cmd= make_cmd(name,mp,ip,emissive,scene_data[i]["translate"],scene_data[i]["rotate"],scene_data[i]["scale"][0],roughness)
        cmd_list.append(cmd)
        i = i + 1
    
    all_cmds = "\n\n".join(cmd_list)
    dst_path = save_pyscene_with_worker_id(scene_path,worker_id,all_cmds)
    
    testbed.load_scene(dst_path)
    print(scene_path)
    print(f"[Worker {worker_id}] Testbed ready.")
    cnt = 0
    while True:
        task = task_queue.get()

        if task == "STOP":
            print(f"[Worker {worker_id}] Stopping.")
            break
        if len(task) == 2:
            i, output_path = task  # task payload
            result = generate_random_material_from_config_step(configs)
            for mat_name, mat_props in result.items():
                if "roughness" in mat_props:
                    testbed.scene.set_roughness(mat_name, mat_props["roughness"])
                if "baseColor" in mat_props:
                    testbed.scene.set_basecolor(mat_name, mat_props["baseColor"])
            emissive_idx = random.randint(0, 99)
            emissive_name = emissive_list[emissive_idx]
            cam = testbed.scene.camera
            ##ps = [random.uniform(-0.5,0.5),random.uniform(-0.7,0.7),random.uniform(-0.5,3.5)]
            ps = [0,0.2,2]
            cam.position = np.array(ps)
            tt = np.array([0,-0.6,0])
            # tt[0] = tt[0] + random.uniform(-0.4,0.4)
            # tt[2] = tt[2] + random.uniform(-0.4,0.4)
            cam.target = tt
            testbed.scene.setMaterialSlotByTexturePath("back",emissive_path.format(emissive_name),"Emissive")
            scene_data = generate_scene_data(impostor_list,model_path,100)

            for data in scene_data:
                testbed.scene.updateNodeTransformName(data["name"],data["scale"],data["translate"],data["rotate"])
            # rand_state = {
            #     "roughness": float(r),
            #     "camera": {
            #         "position": list(f3_to_numpy(cam.position)),
            #         "target": list(f3_to_numpy(cam.target)),
            #     }
            # }
        else:
            i, output_path, rand_state = task
            testbed.scene.set_roughness("Wall", rand_state["roughness"] )
            cam = testbed.scene.camera
            cam.position = np.array(rand_state["camera"]["position"])
            cam.target = np.array(rand_state["camera"]["target"])
        level_output_path = output_path + f"{i}/"
        os.makedirs(level_output_path, exist_ok=True)

        # === 渲染逻辑（你原来的 render 内部） ===
        
        node_list = testbed.scene.get_scene_graph()
        
        node_dict = {}
        for node in node_list:
            node_dict[node["name"]] = node
        for key in result.keys():
            node_dict[key]["roughness"] = result[key]["roughness"]
            node_dict[key]["baseColor"] = result[key]["baseColor"]
        camera_data ={}
        camera_data["position"] = list(f3_to_numpy( testbed.scene.camera.position))
        camera_data["forward"] = list(normalize(f3_to_numpy( testbed.scene.camera.target) - f3_to_numpy( testbed.scene.camera.position)))
        # Run render
        testbed.run()

        for name in select_list:
            index = object_data_dict[name]
            print(index)
            testbed.capture_output(
                level_output_path + f'{name}_{i:05d}.exr',
                index
            )

        print(f"[Worker {worker_id}] Finished frame {i}")
        with open(level_output_path + "node.json","w") as f:
            json.dump(node_dict, f, indent=4, ensure_ascii=False)
        with open(level_output_path + "camera.json","w") as f:
            json.dump(camera_data, f, indent=4, ensure_ascii=False)
        result_queue.put((i))
def start_render_farm(resolution, scene_path, output_path,
                      object_data_dict, select_list,
                      num_workers=8, num_frames=500,impostor_list = [],emissive_list = []):

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # 启动 Worker
    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(wid, resolution, scene_path,
                  object_data_dict, select_list,
                  task_queue,result_queue,impostor_list,emissive_list)
        )
        p.start()
        workers.append(p)
    
    # 任务分配（Round Robin）
    for i in range(num_frames):
        task_queue.put((i, output_path))
 
    # 全部任务完成后，让 Worker 停止
    for _ in workers:
        task_queue.put("STOP")

    # 等待 Worker 完成
    for p in workers:
        p.join()

    print("=== All rendering tasks completed ===")


            
def main():
    scene_name = "cornel_box"
    scene = r"{}.pyscene".format(scene_name)
    #impostor_list = ["Bunny","dragon","box","box2",]
    impostor_list = []
    emissive_list = os.listdir(r"H:\Falcor\emissive_crop/")

    datasets_path = r"H:\Falcor\datasets\renderdata/" + scene_name + "/"

    if os.path.exists(datasets_path) == False:
        os.makedirs(datasets_path)

    # Create device and setup renderer.
    start_render_farm(
        resolution=[1920,1080],
        scene_path=scene_name,
        output_path=datasets_path,
        object_data_dict=object_key_dict,
        select_list=sellect_list,
        num_workers=2,
        num_frames=200,
        impostor_list=impostor_list,
        emissive_list=emissive_list
    )

if __name__ == "__main__":
    main()
