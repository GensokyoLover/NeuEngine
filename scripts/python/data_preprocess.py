
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from futils import  *
import os
import json
from utils import * 
import cv2
import torch
from impostor import *
def renderdata_preprocess(path,test=False):
    roughness = torch.round(torch.arange(0.0, 1.0 + 1e-9, 0.01) * 100) / 100
    key_list = ["position","albedo","specular","normal","roughness","depth","emission","AccumulatePassoutput","view","raypos","mind","reflect","idepth0","idepth1","idepth2","idirection0","idirection1","idirection2","uv0","uv1","uv2"]
    
    theta95_deg = ggx_theta_from_roughness(roughness, alpha_is_roughness_sq=True, degrees=True,energy_clamp=75).cuda()
    print(theta95_deg)
    MAX_BUCKET_SIZE = 64 * 64  # 65536
    SAVE_ROOT =path + "train/"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    file_list = os.listdir(path)
    final_bucket = {}
    bucket_save_count = {}  
    cnt = 0
    for scene_str in file_list:
        if scene_str == "train":
            continue
        scene_id = int(scene_str)
        folder = os.path.join(path, f"{scene_id}")
        sample = {}

        for key in  key_list:
            exr_path = os.path.join(folder, f"{key}_{scene_id:05d}.exr")

            data = pyexr.read(exr_path)[..., : buffer_channel[key]]

            sample[key] = torch.Tensor(data).cuda()  # shape: (H,W,C)
        sample["sampleView"] = torch.cat([sample["idirection0"][...,3:4],sample["idirection1"][...,3:4],sample["idirection2"][...,3:4]],dim=-1)
        #pyexr.write("H:/Falcor/media/inv_rendering_scenes/object_level_config/Bunny/{}roughness.exr".format(scene_id),sample["roughness"].cpu().numpy()) 
        sample["mask"] = (sample["position"].sum(dim=-1,keepdim=True)!=0)  & (sample["normal"][...,2:3]<0.01)  & (sample["emission"].sum(dim=-1,keepdim=True)==0) & (sample["idepth0"][...,0:1] != 0) & (sample["idepth1"][...,0:1] != 0) & (sample["idepth2"][...,0:1] != 0)
        #sample["mask"] = (sample["position"].sum(dim=-1,keepdim=True)!=0) & (sample["roughness"] > 0.0001) & (sample["normal"][...,2:3]<0.1)
        if sample["mask"].sum() < 1:
            print("continue ",scene_id)
            continue
        json_path = os.path.join(folder, "node.json")
        with open(json_path, "r") as f:
            node = json.load(f)
        half_angle_deg = theta95_deg[(sample["roughness"].reshape(-1)*100).long()].reshape(-1)
        sample["node"] = node  # dict，不转 GPU
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

        buckets = group_by_view_triplet_with_roughness_fast(sample)
        #buckets2 = group_by_view_triplet_with_roughness(sample)

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
            print(triplet, final_bucket[triplet]["AccumulatePassoutput"].shape[0])
            # 检查长度是否超过阈值
            N = final_bucket[triplet]["AccumulatePassoutput"].shape[0]

            while N > MAX_BUCKET_SIZE:

                # 截取前 MAX_BUCKET_SIZE 条保存
                chunk_data = {k: v[:MAX_BUCKET_SIZE] for k, v in final_bucket[triplet].items()}
                save_bucket_chunk(SAVE_ROOT,scene_id, triplet, chunk_data, bucket_save_count[triplet])
                bucket_save_count[triplet] += 1

                # 删除已保存的部分，保留剩余
                final_bucket[triplet] = {k: v[MAX_BUCKET_SIZE:] for k, v in final_bucket[triplet].items()}
                N = final_bucket[triplet]["AccumulatePassoutput"].shape[0]

    return sample,scene_id

import os
import json
import torch
import cv2
import pyexr
import pickle
import zstandard as zstd  # 需要 pip install zstandard

def load_impostor2(path):
    """
    加载 Impostor 数据，并可选择将其保存为 pkl.zst 格式。
    
    Args:
        name (str): 场景名称
        level (int): 等级
        save_dir (str, optional): 如果提供路径，将把生成的对象保存为 .pkl.zst 文件到该目录
    """
    # 原始路径逻辑保持不变

    
    impostor = ImpostorPT()
    
    # --- 2. 原始加载逻辑 ---
    print(f"Loading raw files from {path}...")
    
    with open(path + "basic_info.json", "r") as file:
        basic_info = json.load(file)
    impostor.radius = basic_info["radius"]
    impostor.centerWS = torch.tensor(basic_info["centorWS"], dtype=torch.float32).cuda()
    
    with open(path + "faces.json", "r") as file:
        faces_info = json.load(file)
    impostor.cFace = torch.tensor(faces_info, dtype=torch.int32).cuda()
    
    with open(path + "forward.json", "r") as file:
        forward_info = json.load(file)
    impostor.cForward = torch.tensor(forward_info, dtype=torch.float32).cuda()
    
    with open(path + "up.json", "r") as file:
        up_info = json.load(file)
    impostor.cUp = torch.tensor(up_info, dtype=torch.float32).cuda()
    
    with open(path + "right.json", "r") as file:
        right_info = json.load(file)
    impostor.cRight = torch.tensor(right_info, dtype=torch.float32).cuda()
    
    with open(path + "position.json", "r") as file:
        position_info = json.load(file)
    impostor.cPosition = torch.tensor(position_info, dtype=torch.float32).cuda()
    
    with open(path + "radius.json", "r") as file:
        radius_info = json.load(file)
    impostor.cRadius = torch.tensor(radius_info, dtype=torch.float32).cuda().unsqueeze(-1)
    
    # 读取 lookup_uint16.png
    texFaceIndex = cv2.imread(path + "lookup_uint16.png", cv2.IMREAD_UNCHANGED)
    impostor.texFaceIndex = torch.tensor(texFaceIndex, dtype=torch.int32).cuda()
    
    impostor.texDict = {}
    scale =1 / impostor.cRadius[0]
    # 初始化张量
    for key in ["albedo", "specular", "depth", "emission", "normal", "position", "roughness", "view", "raypos"]:
        # 注意：这里假设 buffer_channel 是外部定义的全局变量
        impostor.texDict[key] = torch.zeros((42, 512, 512, buffer_channel[key])).cuda()
    for key in ["depth", "position","raypos"]:
        impostor.texDict[key] = impostor.texDict[key] * scale
    subdirs = [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    
    for file_id_str in subdirs:
        # 优化循环：只遍历存在的子目录
        for name_key in ["albedo", "specular", "depth", "emission", "normal", "position", "roughness", "view", "raypos"]:
            id = int(file_id_str)
            image_path = path + file_id_str + "/" + name_key + f"_{id:05d}.exr"
            
            # 读取 EXR 并转为 Tensor
            data = pyexr.read(image_path)[:, :, :buffer_channel[name_key]]
            impostor.texDict[name_key][id, :, :, :] = torch.tensor(data, dtype=torch.float32).cuda()

    # --- 3. 新增：保存为 pkl.zst ---
 

   
    filename = f"impostor.pkl.zst"
    output_path = os.path.join(path, filename)
    
    print(f"Compressing and saving to {output_path}...")
    
    # 使用 zstandard 压缩上下文
    cctx = zstd.ZstdCompressor(level=3) # level 3 是速度和压缩率的良好平衡
    packed = pickle.dumps(impostor)
    with open(output_path, "wb") as f:
        f.write(cctx.compress(packed))
            
    print("Save complete.")

    return impostor


if __name__ == "__main__":
    path = r"H:\Falcor\datasets\renderdata/flame/"
    renderdata_preprocess(path,test=False)
    # load_impostor2(r"H:\Falcor\datasets\impostor\flame\level1/")