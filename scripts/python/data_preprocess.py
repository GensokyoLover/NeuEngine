
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

import os
import json
import torch
import pyexr
import pickle
import zstandard as zstd
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# 假设 buffer_channel 和 ggx_theta_from_roughness 定义在外部或此处
# buffer_channel = { ... } 

def process_one_scene(scene_str, path, key_list, theta95_deg_cpu, test, buffer_channel):
    """
    运行在子进程中的函数。
    注意：这里全部使用 CPU Tensor，不要调用 .cuda()
    """
    try:
        if scene_str == "train":
            return None
            
        scene_id = int(scene_str)
        folder = os.path.join(path, f"{scene_id}")
        sample = {}

        # 1. 读取数据 (I/O)
        for key in key_list:
            exr_path = os.path.join(folder, f"{key}_{scene_id:05d}.exr")
            # 这里的 pyexr 读取后直接转为 CPU Tensor
            data = pyexr.read(exr_path)[..., : buffer_channel[key]]
            sample[key] = torch.from_numpy(data).float() # 保持在 CPU

        # 2. 构造 sampleView
        sample["sampleView"] = torch.cat([
            sample["idirection0"][..., 3:4],
            sample["idirection1"][..., 3:4],
            sample["idirection2"][..., 3:4]
        ], dim=-1)

        # 3. 计算 Mask (CPU计算)
        # 注意：这里的所有逻辑都必须能在 CPU 上运行
        mask = (sample["AccumulatePassoutput"].sum(dim=-1, keepdim=True) != 0) & \
               (sample["position"].sum(dim=-1, keepdim=True) != 0) & \
               (sample["normal"][..., 2:3] < 0.01) & \
               (sample["emission"].sum(dim=-1, keepdim=True) == 0) & \
               (sample["idepth0"][..., 0:1] != 0) & \
               (sample["idepth1"][..., 0:1] != 0) & \
               (sample["idepth2"][..., 0:1] != 0)
        
        sample["mask"] = mask

        if sample["mask"].sum() < 1:
            return None # 此时没有有效数据

        # 4. 读取 JSON
        json_path = os.path.join(folder, "node.json")
        with open(json_path, "r") as f:
            node = json.load(f)
        sample["node"] = node

        # 计算 half_angle (使用传入的 CPU 版 theta95)
        # 注意：确保 theta95_deg_cpu 是 CPU tensor
        half_angle_deg = theta95_deg_cpu[(sample["roughness"].reshape(-1)*100).long()].reshape(-1)
        
        # --- TEST 模式分支 ---
        if test:
            # Test 模式下，子进程直接压缩并返回二进制数据，或者直接保存
            # 为了减少 IPC 通信压力，建议子进程直接保存文件，只返回文件名或计数
            # 但你的逻辑里 cnt 是全局的，这里需要特殊处理。
            # 策略：返回处理好的 sample 对象，由主进程写入，或者返回压缩好的 bytes
            enc = zstd.ZstdCompressor(level=3)
            packed = pickle.dumps(sample)
            compressed_data = enc.compress(packed)
            return ("TEST_DATA", compressed_data)

        # --- TRAIN 模式分支 ---
        # 调用分组函数
        # ！！！关键点：确保这个函数 group_by_view_triplet_with_roughness_fast 能接受 CPU Tensor
        buckets = group_by_view_triplet_with_roughness_fast(sample)
        
        # 返回 buckets 给主进程合并
        return ("TRAIN_BUCKETS", buckets, scene_id)

    except Exception as e:
        print(f"Error processing scene {scene_str}: {e}")
        return None

def renderdata_preprocess2(path, test=False):
    # 1. 准备常量和配置
    roughness = torch.round(torch.arange(0.0, 1.0 + 1e-9, 0.01) * 100) / 100
    key_list = ["position", "albedo", "specular", "normal", "roughness", "depth", "emission", 
                "AccumulatePassoutput", "view", "raypos", "mind", "reflect", "idepth0", "idepth1", 
                "idepth2", "idirection0", "idirection1", "idirection2", "uv0", "uv1", "uv2"]
    
    # 计算 theta95，并确保它在 CPU 上，以便传递给子进程
    # 注意：这里去掉了 .cuda()，如果计算过程依赖 cuda，算完后要 .cpu()
    theta95_deg = ggx_theta_from_roughness(roughness, alpha_is_roughness_sq=True, degrees=True, energy_clamp=75)
    if theta95_deg.is_cuda:
        theta95_deg = theta95_deg.cpu()

    MAX_BUCKET_SIZE = 256 * 256
    SAVE_ROOT = path + "train2/"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if test:
        os.makedirs(os.path.join(path, "test"), exist_ok=True)

    file_list = [f for f in os.listdir(path) if f != "train" and f != "test"]
    file_list = file_list[:]
    final_bucket = {}
    bucket_save_count = {}
    cnt = 0

    # 2. 配置多进程池
    # 这里的 processes 根据你的 CPU 核心数调整，建议设置为 核心数-2 或 8-16 左右
    num_workers = min(mp.cpu_count(), 16) 
    pool = mp.Pool(processes=num_workers)

    # 固定部分参数
    worker_func = partial(process_one_scene, path=path, key_list=key_list, 
                          theta95_deg_cpu=theta95_deg, test=test, buffer_channel=buffer_channel)

    print(f"Starting multiprocessing with {num_workers} workers...")

    # 3. 迭代处理结果
    # imap_unordered 会让子进程处理完一个就立刻返回一个，不需要等待所有结束
    for result in tqdm(pool.imap_unordered(worker_func, file_list), total=len(file_list)):
        if result is None:
            continue
        
        type_tag = result[0]

        if type_tag == "TEST_DATA":
            # Test 模式：保存数据
            compressed_data = result[1]
            fn = f"{cnt}.pkl.zst"
            full_path = os.path.join(path, "test", fn)
            with open(full_path, "wb") as f:
                f.write(compressed_data)
            cnt += 1

        elif type_tag == "TRAIN_BUCKETS":
            # Train 模式：合并 Bucket
            new_buckets, scene_id = result[1], result[2]
            
            for triplet, bucket_data in new_buckets.items():
                # 初始化 final bucket
                if triplet not in final_bucket:
                    # 注意：如果子进程返回的是 CPU tensor，这里可以视情况 .cuda() 
                    # 如果显存够大，可以放 GPU；否则建议先在 CPU 上拼，保存时再处理
                    final_bucket[triplet] = {k: v for k, v in bucket_data.items()}
                    bucket_save_count[triplet] = 0
                else:
                    # 拼接
                    for key in final_bucket[triplet]:
                        final_bucket[triplet][key] = torch.cat([
                            final_bucket[triplet][key],
                            bucket_data[key]
                        ], dim=0)
                
                # 检查阈值并保存
                N = final_bucket[triplet]["AccumulatePassoutput"].shape[0]
                while N > MAX_BUCKET_SIZE:
                    print(f"Saving chunk for triplet {triplet}, scene {scene_id}...")
                    
                    # 截取
                    chunk_data = {k: v[:MAX_BUCKET_SIZE] for k, v in final_bucket[triplet].items()}
                    
                    # 保存 (save_bucket_chunk 需要自己保证能处理 CPU tensor 或内部转 cuda)
                    save_bucket_chunk(SAVE_ROOT, scene_id, triplet, chunk_data, bucket_save_count[triplet])
                    bucket_save_count[triplet] += 1

                    # 保留剩余
                    final_bucket[triplet] = {k: v[MAX_BUCKET_SIZE:] for k, v in final_bucket[triplet].items()}
                    N = final_bucket[triplet]["AccumulatePassoutput"].shape[0]

    pool.close()
    pool.join()
    print("Preprocessing finished.")
    # 注意：函数结束时可能 final_bucket 里还有剩余数据没存，视你的业务逻辑决定是否要 flush 剩余数据

def renderdata_preprocess(path,test=False):
    roughness = torch.round(torch.arange(0.0, 1.0 + 1e-9, 0.01) * 100) / 100
    key_list = ["position","albedo","specular","normal","roughness","depth","emission","AccumulatePassoutput","view","raypos","mind","reflect"]
    for i in range(2):
        key_list.append()
    
    theta95_deg = ggx_theta_from_roughness(roughness, alpha_is_roughness_sq=True, degrees=True,energy_clamp=75).cuda()
    print(theta95_deg)
    MAX_BUCKET_SIZE = 256 * 256  # 65536
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
        sample["mask"] =(sample["AccumulatePassoutput"].sum(dim=-1,keepdim=True)!=0) & (sample["position"].sum(dim=-1,keepdim=True)!=0)  & (sample["normal"][...,2:3]<0.01)  & (sample["emission"].sum(dim=-1,keepdim=True)==0) & (sample["idepth0"][...,0:1] != 0) & (sample["idepth1"][...,0:1] != 0) & (sample["idepth2"][...,0:1] != 0)
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
    impostor.cPosition = impostor.cPosition * scale
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
    path = r"H:\Falcor\datasets\renderdata/cornel_box/"
    #renderdata_preprocess(path,test=False)
    renderdata_preprocess2(path,test=False)
    # load_impostor2(r"H:\Falcor\datasets\impostor\flame\level1/")