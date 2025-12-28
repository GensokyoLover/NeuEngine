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

def render_graph_MinimalPathTracer(testbed):
    g = testbed.create_render_graph("MinimalPathTracer")
    g.create_pass("AccumulatePass","AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.create_pass("ToneMapper","ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.create_pass("MinimalPathTracer","MinimalPathTracer", {'maxBounces': 3})
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

import numpy as np

import multiprocessing as mp
def worker_process(worker_id, resolution, scene_path,
                   object_data_dict, select_list,
                   task_queue: mp.Queue):
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
        spp=256
    )
    render_graph_MinimalPathTracer(testbed)
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
    # Worker 主循环：一直等待任务
    while True:
        task = task_queue.get()

        if task == "STOP":
            print(f"[Worker {worker_id}] Stopping.")
            break

        i, output_path = task  # task payload
        level_output_path = output_path + f"{i}/"
        os.makedirs(level_output_path, exist_ok=True)


        print(r)
        testbed.scene.set_roughness("Wall", 0)

        cam = testbed.scene.camera
        cam.position = np.random.uniform(-0.3, 0.3, size=[3]) + np.array([0.0, 0.25, 1.2])
        cam.target = np.array([0.0, -0.5, 0.0]) + np.random.uniform(-0.3, 0.3, size=[3])
        node_list = testbed.scene.get_scene_graph()
        node_dict = {}
        for node in node_list:
            node_dict[node["name"]] = node
        print(node_dict["Light"]["transform"])
        camera_data ={}
        camera_data["position"] = list(f3_to_numpy( testbed.scene.camera.position))
        camera_data["forward"] = list(normalize(f3_to_numpy( testbed.scene.camera.target) - f3_to_numpy( testbed.scene.camera.position)))
        # Run render
        testbed.run()

        for name in select_list:
            index = object_data_dict[name]
            testbed.capture_output(
                level_output_path + f'{name}_{i:05d}.exr',
                index
            )

        print(f"[Worker {worker_id}] Finished frame {i}")
        with open(level_output_path + "node.json","w") as f:
            json.dump(node_dict, f, indent=4, ensure_ascii=False)
        with open(level_output_path + "camera.json","w") as f:
            json.dump(camera_data, f, indent=4, ensure_ascii=False)

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


object_data_list = ["color","position","albedo","specular","normal","roughness","depth","AccumulatePassoutput","emission","view","raypos"]
object_key_dict = {name: i for i, name in enumerate(object_data_list)}
sellect_list = ["depth","view","raypos"]
#object_data_list = ["color","position","albedo","specular","normal","roughness","depth","AccumulatePassoutput"]
def pack_object_data(path,camera_resolution,direction_resolution):
    data = {}
    for name in object_data_list:
        data[name] = pyexr.read(path + name + ".exr").reshape(camera_resolution,direction_resolution,camera_resolution,direction_resolution,-1).transpose(2,0,1,3,4)
    save_compressed_pickle(data,path + "light.pkl.zst")


import os
import pyexr
import numpy as np

def convert_exr_to_rgba(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".exr"):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        try:
            # 1️⃣ 读取 EXR 文件
            img = pyexr.open(in_path)
            arr = img.get()  # shape: (H, W, C)

            # 2️⃣ 判断通道数
            H, W = arr.shape[:2]
            C = arr.shape[2] if arr.ndim == 3 else 1

            # 3️⃣ 生成 RGBA
            rgba = np.zeros((H, W, 4), dtype=np.float32)

            if C >= 3:
                rgba[:, :, :3] = arr[:, :, :3]
            elif C == 1:
                rgba[:, :, 0] = arr[:, :, 0]
            # alpha 补 0（默认值已是 0）

            # 4️⃣ 保存为 RGBA EXR
            pyexr.write(out_path, rgba)
            print(f"[OK] {fname} -> RGBA ({W}x{H}) saved.")
        except Exception as e:
            print(f"[Error] {fname}: {e}")


def main():
    # get_bounding_box(r'E:\TOG\Falcor\media\test_scenes\meshes\bunny_centered.obj')
    # exit()
    # lookup_loaded = cv2.imread(r"H:\Falcor\scripts\python\scenes\onlybunny\level0/" + "lookup_uint16.png", cv2.IMREAD_UNCHANGED)
    # print(lookup_loaded[0,0])
    # print(lookup_loaded[-1,-1])
    # lookup_loaded = lookup_loaded.astype(np.float32)
    # pyexr.write("./lookup.exr",lookup_loaded)

    finest_resolution = 512
    folder_path = r"H:\Falcor\media\inv_rendering_scenes\object_level_config/"
    file_list = os.listdir(r"H:\Falcor\media\inv_rendering_scenes\object_level_config/")
    for file in file_list: 
        if file.split(".")[-1] != "pyscene":
            continue
        # if (file.split(".")[0] + "level3") in file_list:
        #     continue
        scene_path = folder_path + file
        
        output_folder = scene_path.replace('.pyscene','')
        os.makedirs(output_folder, exist_ok=True)
        # Create device and setup renderer.
        device = falcor.Device(type=falcor.DeviceType.Vulkan, gpu=0, enable_debug_layer=False)
        testbed = falcor.Testbed(width=finest_resolution, height=finest_resolution, create_window=False, device=device,spp=16)
        render_graph_MinimalPathTracer(testbed)
    
        generate_impostor_by_falcor(finest_resolution,testbed,scene_path,output_folder,object_key_dict,sellect_list)
        # start_render_farm(finest_resolution,testbed,scene_path,output_folder,object_key_dict,sellect_list)
        task_queue = mp.Queue()

        # 启动 Worker
        workers = []
        for wid in range(8):
            p = mp.Process(
                target=worker_process,
                args=(wid, 512, scene_path,
                    object_key_dict, sellect_list,
                    task_queue)
            )
            p.start()
            workers.append(p)
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
            task_queue.put((i, output_path))
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
        # 任务分配（Round Robin）
    
            

        # 全部任务完成后，让 Worker 停止
        for _ in workers:
            task_queue.put("STOP")

        # 等待 Worker 完成
        for p in workers:
            p.join()

        print("=== All rendering tasks completed ===")
        



# if __name__ == "__main__":
#     for i in range(4):
#         input_dir = r"H:\Falcor\scripts\python\scenes\onlybunny_1024_scale\level{}".format(i)        # 🟩 修改：输入目录
#         output_dir = r"H:\Falcor\scripts\python\scenes\onlybunny_1024_scale\level{}".format(i)    # 🟩 修改：输出目录
#         convert_exr_to_rgba(input_dir, output_dir)
if __name__ == "__main__":
    main()
