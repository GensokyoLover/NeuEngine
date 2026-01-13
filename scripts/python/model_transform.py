import os
import trimesh
import numpy as np
import math

import trimesh
import numpy as np
import os
import glob

import os
import trimesh
import numpy as np
from pathlib import Path

def process_single_obj(input_path, output_path):
    """
    核心处理函数：加载 -> 居中 -> 缩放 -> 保存
    """
    try:
        # process=False 关键：防止自动合并顶点，保护法线数据
        mesh = trimesh.load(input_path, process=False, force='mesh')
        
        # 获取顶点
        vertices = mesh.vertices
        if len(vertices) == 0:
            print(f"⚠️  Skipping empty mesh: {input_path}")
            return False

        # 1. 计算原始包围盒
        min_vals = vertices.min(axis=0)
        max_vals = vertices.max(axis=0)
        
        # 2. 计算位移 (Center)
        center = (min_vals + max_vals) / 2.0
        
        # 3. 计算缩放 (Scale)
        extents = max_vals - min_vals
        diagonal_length = np.linalg.norm(max_vals - min_vals)
        
        if diagonal_length <= 1e-6:
            print(f"⚠️  Skipping zero-size mesh: {input_path}")
            return False

        scale_factor = 2.0 / diagonal_length

        # 4. 应用变换 (P_new = (P - Center) * Scale)
        # 仅修改顶点位置，绝对不触碰 normals
        mesh.vertices = (vertices - center) * scale_factor

        # 5. 导出
        # include_normals=True 确保写入原始法线
        mesh.export(output_path, include_normals=True)
        return True

    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")
        return False

def batch_process(input_dir, output_dir):
    """
    批量遍历目录
    """
    # 确保路径存在
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有 .obj 文件
    obj_files = list(input_path.glob("*.obj"))
    total_files = len(obj_files)
    
    print(f"Found {total_files} .obj files in '{input_dir}'")
    print("-" * 40)

    count = 0
    for i, file_path in enumerate(obj_files):
        # 构建输出文件名
        save_path = output_path / file_path.name
        
        print(f"[{i+1}/{total_files}] Processing: {file_path.name} ... ", end="", flush=True)
        
        if process_single_obj(str(file_path), str(save_path)):
            print("Done.")
            count += 1
        else:
            print("Failed.")

    print("-" * 40)
    print(f"Batch processing complete. Successfully processed {count}/{total_files} files.")
    print(f"Output directory: {output_path.resolve()}")


# --- 使用示例 ---
# input_folder = r"H:\Falcor\media\objaverse_glb/"
# output_folder = r"H:\Falcor\media\objaverse_glb_normalized/"
# # 现在的 1.0 代表对角线长度为 1.0
# target_diag = 2.0  

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# for glb_file in glob.glob(os.path.join(input_folder, "*.glb")):
#     process_glb_diagonal(glb_file, os.path.join(output_folder, os.path.basename(glb_file)), target_diag)

def normalize_obj_to_aabb_radius_1(input_dir):
    assert os.path.isdir(input_dir), f"{input_dir} 不是一个目录"

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".obj"):
            continue

        in_path = os.path.join(input_dir, fname)
        name, ext = os.path.splitext(fname)
        out_path = os.path.join(input_dir, f"{name}.obj")

        print(f"[Processing] {fname}")

        # 1️⃣ 读取模型
        mesh = trimesh.load(in_path, force='mesh')

        if mesh.is_empty:
            print(f"  ⚠️ 跳过空模型: {fname}")
            continue

        # 2️⃣ 计算 AABB
        min_bound, max_bound = mesh.bounds
        extents = max_bound - min_bound
        L  = math.sqrt(extents[0]**2 + extents[1]**2 + extents[2]**2)   # 对角线长度

        if L <= 0:
            print(f"  ⚠️ 非法 AABB（L=0）: {fname}")
            continue

        # 3️⃣ 平移到 AABB 中心
        center = (min_bound + max_bound) * 0.5
        mesh.apply_translation(-center)

        # 4️⃣ 缩放到最长半径 = 1（最长边 = 2）
        scale = 2.0 / L
        mesh.apply_scale(scale)

        # 5️⃣ 导出
        mesh.export(out_path)

        print(f"  ✅ 输出: {out_path}")

    print("=== 全部处理完成 ===")


if __name__ == "__main__":
  
    INPUT_FOLDER = r"H:\Falcor/model/"      # 原始 OBJ 文件夹
    OUTPUT_FOLDER = r"H:\Falcor/model/" # 处理后存放文件夹
    
    batch_process(INPUT_FOLDER, OUTPUT_FOLDER)
