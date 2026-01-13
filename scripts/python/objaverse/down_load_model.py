import objaverse
import os
import random
import shutil
import concurrent.futures
from tqdm import tqdm  # 进度条库

def process_single_model(uid, output_dir):
    """
    单个模型的处理函数：下载 -> 移动 -> 清理
    """
    target_filename = f"{uid}.glb"
    target_path = os.path.join(output_dir, target_filename)

    # 1. 检查是否已存在（断点续传）
    if os.path.exists(target_path):
        return True # 已经有了，跳过

    try:
        # 2. 下载单个模型
        # download_processes=1 确保我们在外部控制并发，而不是库内部
        objects = objaverse.load_objects(uids=[uid], download_processes=1)
        
        # 3. 获取缓存路径
        if not objects:
            return False
        
        cache_path = objects[uid]
        
        # 4. 移动文件 (使用 move 而不是 copy，可以节省C盘缓存空间)
        shutil.move(cache_path, target_path)
        
        return True
    except Exception as e:
        # print(f"Error {uid}: {e}") # 出错时不打印太多信息以免刷屏
        return False

def download_glb_models_concurrent(
    num_models=4000, 
    output_dir="H:/Falcor/media/objaverse_glb",
    max_workers=16 # 建议设置为 8-32 之间，取决于你的网速和CPU
):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"1. Loading UIDs list...")
    all_uids = objaverse.load_uids()
    
    # 2. 随机选取 UID
    # 如果你想确保每次运行都尝试下载新的，可以先扫描文件夹排除已下载的UID
    existing_files = {f.split('.')[0] for f in os.listdir(output_dir) if f.endswith('.glb')}
    available_uids = [u for u in all_uids if u not in existing_files]
    
    print(f"   已存在: {len(existing_files)}")
    print(f"   剩余可用: {len(available_uids)}")
    
    if len(available_uids) < num_models:
        print("警告: 剩余可用模型不足请求数量，将下载所有剩余模型。")
        selected_uids = available_uids
    else:
        selected_uids = random.sample(available_uids, num_models)

    print(f"2. Starting concurrent download for {len(selected_uids)} models with {max_workers} threads...")

    # 3. 多线程执行
    success_count = 0
    
    # 使用 ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        # future_to_uid 字典用于跟踪任务
        future_to_uid = {
            executor.submit(process_single_model, uid, output_dir): uid 
            for uid in selected_uids
        }
        
        # 使用 tqdm 显示进度条
        for future in tqdm(concurrent.futures.as_completed(future_to_uid), total=len(selected_uids), unit="model"):
            uid = future_to_uid[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as exc:
                pass

    print(f"\nDone! Successfully downloaded: {success_count}/{len(selected_uids)}")
    print(f"Models are located in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 建议先设置较小的数量测试一下，比如 50，确认速度后再跑 4000
    download_glb_models_concurrent(
        num_models=4000, 
        output_dir="H:/Falcor/media/objaverse_glb",
        max_workers=20 # 设置为20个并发下载
    )
