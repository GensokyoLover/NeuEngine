import cv2
import os

def extract_flipped_frames(video_path, output_folder, image_format="png"):
    """
    将视频的每一帧提取并保存为上下颠倒（垂直翻转）的图片。

    :param video_path: 输入MP4视频文件的完整路径。
    :param output_folder: 保存提取图片的目录路径。
    :param image_format: 保存图片的格式（例如："png", "jpg"）。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建了输出文件夹: {output_folder}")

    # 1. 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件: {video_path}")
        return

    frame_count = 0
    # 定义翻转代码：0 表示垂直翻转（上下颠倒）
    FLIP_CODE = 0 

    print(f"正在处理视频: {video_path}...")
    
    # 2. 逐帧读取、翻转和保存
    while True:
        ret, frame = cap.read()

        # 如果 ret 为 False，表示视频结束或读取失败
        if not ret:
            break

        # **************** 关键修改点 ****************
        # 3. 对当前帧进行垂直翻转 (上下颠倒)
        flipped_frame = cv2.flip(frame, FLIP_CODE)
        # *****************************************

        # 4. 构造输出文件名
        # 格式为 frame_00000.png, frame_00001.png, ...
        frame_filename = f"frame_{frame_count:05d}.{image_format}"
        output_file_path = os.path.join(output_folder, frame_filename)

        # 5. 保存翻转后的帧为图片文件
        cv2.imwrite(output_file_path, flipped_frame)

        frame_count += 1

    # 6. 释放资源
    cap.release()
    
    print("-" * 30)
    print(f"✅ 成功提取并翻转 {frame_count} 帧。")
    print(f"图片保存在: {output_folder}")

# --- 配置您的路径 ---
# 确保将以下路径替换为您实际的文件和目录
INPUT_VIDEO = "E:\\QQ20251110-153438-HD.mp4" 
OUTPUT_DIR = "output_flipped_frames"

# 执行函数
extract_flipped_frames(INPUT_VIDEO, OUTPUT_DIR)