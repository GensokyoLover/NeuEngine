import sys
import os


os.environ["PATH"]      = r"H:/Falcor\build\windows-vs2022\bin\Debug" + os.environ["PATH"]
sys.path.append( r"H:/Falcor\build\windows-vs2022\bin\Debug/python")
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
def render_graph_MinimalPathTracer(testbed):
    g = testbed.create_render_graph("MinimalPathTracer")
    g.create_pass("AccumulatePass", "AccumulatePass",{'enabled': True, 'precisionMode': 'Single'})
    
    g.create_pass("ToneMapper", "ToneMapper",{'autoExposure': False, 'exposureCompensation': 0.0})
    g.create_pass("MinimalPathTracer","MinimalPathTracer", {'maxBounces': 0})
    
    g.create_pass("VBufferRT","VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16,"useTraceRayInline":False})
    #g.create_pass("AccumulatePass2", "AccumulatePass",{'enabled': True, 'precisionMode': 'Single'})
    #g.create_pass("MinimalPathTracer2", "MinimalPathTracer",{'maxBounces': 0})
    g.create_pass("VBufferRT2","VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16,"useTraceRayInline":True})
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer")
    
    g.addEdge("VBufferRT.viewW", "MinimalPathTracer.viewW")
    g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")
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
    g.markOutput("VBufferRT2.depth0")
    g.markOutput("VBufferRT2.depth1")
    g.markOutput("VBufferRT2.depth2")
    g.markOutput("VBufferRT2.direction0")
    g.markOutput("VBufferRT2.direction1")
    g.markOutput("VBufferRT2.direction2")
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
    # 将数据序列化为 pickle 格式
    pickled_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)

    # 创建 zstd 压缩器
    cctx = zstd.ZstdCompressor()

    # 压缩数据
    compressed_data = cctx.compress(pickled_data)
    #print(file_path)
    # 将压缩后的数据写入文件
    with open(file_path, 'wb') as f:
        f.write(compressed_data)


object_data_list = ["color","position","albedo","specular","normal","roughness","depth","emission","AccumulatePassoutput","view","raypos","mind","reflect","idepth0","idepth1","idepth2","idirection0","idirection1","idirection2"]
#object_data_list = ["idepth0","idepth1","idepth2","idirection0","idirection1","idirection2"]
object_key_dict = {name: i for i, name in enumerate(object_data_list)}
sellect_list = ["albedo","specular","normal","position","view","AccumulatePassoutput","roughness","raypos","depth","emission","mind","reflect","idepth0","idepth1","idepth2","idirection0","idirection1","idirection2"]
#sellect_list = ["albedo","specular","normal","position","view","AccumulatePassoutput","roughness","raypos","depth","emission","mind","reflect"]
#sellect_list = ["idepth0","idepth1","idepth2","idirection0","idirection1","idirection2"]
#object_data_list = ["color","position","albedo","specular","normal","roughness","depth","AccumulatePassoutput"]
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

# def sample_roughness() -> float:
#     p = random.random()
#     if p < 0.2:
#         return _uniform_step01(0.07, 0.10)
#     elif p < 0.4:
#         return _uniform_step01(0.10, 0.20)
#     elif p < 0.6:
#         return _uniform_step01(0.20, 0.40)
#     elif p < 0.8:
#         return _uniform_step01(0.40, 0.60)
#     else:
#         return _uniform_step01(0.60, 1.00)

def sample_roughness() -> float:

    return _uniform_step01(0.01, 0.15)

import multiprocessing as mp
def worker_process(worker_id, resolution, scene_path,
                   object_data_dict, select_list,
                   task_queue: mp.Queue,result_queue: mp.Queue):
    """
    Worker 常驻进程：保持一个 Testbed 实例
    """
    print(f"[Worker {worker_id}] Initializing testbed...")

    # 只有 worker 进程初始化 Falcor
    device = falcor.Device(
        type=falcor.DeviceType.Vulkan,
        gpu=0,
        enable_debug_layer=False
    )

    testbed = falcor.Testbed(
        width=resolution,
        height=resolution,
        create_window=False,
        device=device,
        spp=500
    )
    render_graph_MinimalPathTracer(testbed)
    testbed.load_scene(scene_path)
    print(scene_path)
    print(f"[Worker {worker_id}] Testbed ready.")
    while True:
        task = task_queue.get()

        if task == "STOP":
            print(f"[Worker {worker_id}] Stopping.")
            break
        if len(task) == 2:
            i, output_path = task  # task payload
            r = sample_roughness()
       
            testbed.scene.set_roughness("Floor", r )
            cam = testbed.scene.camera
            ps = np.random.uniform(-2, 2, size=[3]) + np.array([0.0, 1.0, 0.0])
            ps[1] = ps[1] * 0.4
            cam.position = ps

            cam.target = np.array([0.0,0.0, 0.0]) + np.random.uniform(-0.3, 0.3, size=[3])
            rand_state = {
                "roughness": float(r),
                "camera": {
                    "position": list(f3_to_numpy(cam.position)),
                    "target": list(f3_to_numpy(cam.target)),
                }
            }
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
        result_queue.put((i, rand_state))
def start_render_farm(resolution, scene_path, output_path,
                      object_data_dict, select_list,
                      num_workers=8, num_frames=500):

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # 启动 Worker
    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(wid, resolution, scene_path,
                  object_data_dict, select_list,
                  task_queue,result_queue)
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
def start_render_farm2(
    resolution,
    scene_path,
    occlution_path,
    nolight_path,
    output_path,
    occ_output_path,
    nolight_output_path,
    object_data_dict,
    select_list,
    num_workers=8,
    num_frames=500,
):
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # ===============================
    # 1️⃣ 启动 Worker（只启动一次）
    # ===============================
    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(
                wid,
                resolution,
                scene_path,
                object_data_dict,
                select_list,
                task_queue,
                result_queue,
            ),
        )
        p.start()
        workers.append(p)

    # ======================================================
    # 2️⃣ 第一轮：随机采样 + 渲染 + 记录随机信息
    # ======================================================
    print("=== Pass 1: sampling & rendering ===")

    for i in range(num_frames):
        task_queue.put((i, output_path))

    random_states = {}

    for _ in range(num_frames):
        i, rand_state = result_queue.get()
        random_states[i] = rand_state

    # 保存随机状态（强烈建议）
    os.makedirs(output_path, exist_ok=True)
    rand_state_path = os.path.join(output_path, "random_states.json")
    with open(rand_state_path, "w") as f:
        json.dump(random_states, f, indent=2)

    print(f"[Pass 1] Saved random states to {rand_state_path}")

    # ======================================================
    # 3️⃣ 第二轮：复用随机信息，再完整跑一遍
    # ======================================================
    print("=== Pass 2: replay with fixed random states ===")
    workers = []
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(
                wid,
                resolution,
                nolight_path,
                object_data_dict,
                select_list,
                task_queue,
                result_queue,
            ),
        )
        p.start()
        workers.append(p)
    for i in range(num_frames):
        rand_state = random_states[i]
        task_queue.put((i, nolight_output_path, rand_state))

    # 等待第二轮完成（不再需要 result_queue）
    for _ in range(num_frames):
        result_queue.get()

    # ===============================
    # 4️⃣ 关闭 Worker
    # ===============================
    for _ in workers:
        task_queue.put("STOP")

    for p in workers:
        p.join()

    print("=== All rendering tasks completed (2-pass) ===")
    workers = []
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(
                wid,
                resolution,
                occlution_path,
                object_data_dict,
                select_list,
                task_queue,
                result_queue,
            ),
        )
        p.start()
        workers.append(p)
    for i in range(num_frames):
        rand_state = random_states[i]
        task_queue.put((i, occ_output_path, rand_state))

    # 等待第二轮完成（不再需要 result_queue）
    for _ in range(num_frames):
        result_queue.get()

    # ===============================
    # 4️⃣ 关闭 Worker
    # ===============================
    for _ in workers:
        task_queue.put("STOP")

    for p in workers:
        p.join()

    print("=== All rendering tasks completed (3-pass) ===")

def render(resolution,testbed,scene_path,output_path,object_data_dict,sellect_list):
    testbed.load_scene(scene_path)
    scene = testbed.scene

    final_resolution = resolution
    testbed.scene.camera.nearPlane = 0.001
    basic_info = {}

    
    for i in range(500):
        testbed.resize_frame_buffer(resolution, resolution)

        level_output_path = output_path + "{}/".format(i)
        if not os.path.exists(level_output_path):
            os.makedirs(level_output_path)

        r = sample_roughness()
        testbed.scene.set_roughness("Wall", r )
        testbed.scene.camera.position = np.random.uniform(-0.3,0.3,size=[3]) + np.array([0.0, 0.25, 1.2])
        testbed.scene.camera.target = np.array([0.0, -0.3, 0.0]) + np.random.uniform(-0.3,0.3,size=[3])
        camera_data ={}
        camera_data["position"] = list(f3_to_numpy( testbed.scene.camera.position))
        camera_data["forward"] = list(normalize(f3_to_numpy( testbed.scene.camera.target) - f3_to_numpy( testbed.scene.camera.position)))
        node_list = testbed.scene.get_scene_graph()
        node_dict = {}
        for node in node_list:
            node_dict[node["name"]] = node
        print(node_dict["Light"]["transform"])
        testbed.run()

        for name in sellect_list:
            index = object_data_dict[name]
            tex2 = testbed.capture_output(
                level_output_path + '{}_{:05d}.exr'.format(name, i), index)
        with open(level_output_path + "node.json","w") as f:
            json.dump(node_dict, f, indent=4, ensure_ascii=False)
        with open(level_output_path + "camera.json","w") as f:
            json.dump(camera_data, f, indent=4, ensure_ascii=False)
         
            
def main():
    label ="test"
    scene_path = r'H:\Falcor\scenes\dragon_ref.pyscene'
    resolution = 512
    scale_path = scene_path.replace('.pyscene','') + "_{}/".format(resolution)
    output_path = scene_path.replace('.pyscene','') + "{}/".format(label)
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    # Create device and setup renderer.
    start_render_farm(
        resolution=512,
        scene_path=scene_path,
        output_path=output_path,
        object_data_dict=object_key_dict,
        select_list=sellect_list,
        num_workers=1,
        num_frames=500
    )

if __name__ == "__main__":
    main()
