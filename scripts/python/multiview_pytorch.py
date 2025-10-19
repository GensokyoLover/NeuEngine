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
    g.markOutput("AccumulatePass.output")
    testbed.render_graph = g

def render_graph_ImpostorTracer(testbed):
    g = testbed.create_render_graph("ImpostorTracer")
    g.create_pass("AccumulatePass","AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.create_pass("ToneMapper","ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.create_pass("ImpostorTracer","ImpostorTracer", {'maxBounces': 3})
    g.create_pass("VBufferRT","VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("VBufferRT.vbuffer", "ImpostorTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "ImpostorTracer.viewW")
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


object_data_list = ["color","position","albedo","specular","normal","roughness","depth","AccumulatePassoutput"]
object_key_dict = {name: i for i, name in enumerate(object_data_list)}
sellect_list = ["albedo","depth","normal"]
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
    scene_path = r'./scripts/python/scenes/onlybunny.pyscene'
    output_path = scene_path.replace('.pyscene','') + "/"
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    # Create device and setup renderer.
    device = falcor.Device(type=falcor.DeviceType.Vulkan, gpu=0, enable_debug_layer=False)
    testbed = falcor.Testbed(width=512, height=512, create_window=False, device=device,spp=16)
    render_graph_ImpostorTracer(testbed)
    outputPath = r'H:\\falcor\\image{}x{}\\'

    generate_impostor_by_falcor(testbed,scene_path,output_path,object_key_dict,sellect_list)
    #render(testbed,scene_path,output_path,object_data_list)

# if __name__ == "__main__":
#     for i in range(4):
#         input_dir = r"H:\Falcor\scripts\python\scenes\onlybunny\level{}".format(i)        # 🟩 修改：输入目录
#         output_dir = r"H:\Falcor\scripts\python\scenes\onlybunny\level{}".format(i)    # 🟩 修改：输出目录
#         convert_exr_to_rgba(input_dir, output_dir)
if __name__ == "__main__":
    main()
