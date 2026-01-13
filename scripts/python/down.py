import torch
import torch.nn.functional as F
import os
import pyexr

# =====================================================
# 配置区
# =====================================================
def downsample_depth_minmax(x, fw, fh):
    """
    depth 专用：
    ch0 -> min
    ch1 -> max
    ch2,3 -> mean

    x: (B, W, H, 4)
    return: (B, W/fw, H/fh, 4)
    """
    # (B,W,H,4) -> (B,4,H,W)
    x = x.permute(0, 3, 2, 1).contiguous()

    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    x23 = x[:, 2:4]

    # min pooling
    y0 = -F.max_pool2d(-x0, kernel_size=(fh, fw), stride=(fh, fw))
    # max pooling
    y1 = F.max_pool2d(x1, kernel_size=(fh, fw), stride=(fh, fw))
    # mean pooling
    y23 = F.avg_pool2d(x23, kernel_size=(fh, fw), stride=(fh, fw))

    y = torch.cat([y0, y1, y23], dim=1)

    return y.permute(0, 3, 2, 1).contiguous()
def process_one_file(path, file):
    if not file.isdigit():
        return

    dd = int(file)
    formatted_dd = f"{dd:05d}"

    # ---------- depth ----------
    depth_path = os.path.join(path, file, f"depth_{formatted_dd}.exr")
    data = pyexr.read(depth_path)
    data_tensor = torch.from_numpy(data)

    for level in range(10):
        down_scale = 1 << level
        value = (
            downsample_depth_minmax(
                data_tensor.unsqueeze(0),
                down_scale,
                down_scale
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )

        target_path = os.path.join(path, file, f"{level}depth.exr")
        pyexr.write(target_path, value)

    # ---------- normal ----------
    normal_path = os.path.join(path, file, f"normal_{formatted_dd}.exr")
    normal = pyexr.read(normal_path)
    target_normal_path = os.path.join(path, file, "normal.exr")
    pyexr.write(target_normal_path, normal)
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_impostor(path):
    file_list = os.listdir(path)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for file in file_list:
            futures.append(
                executor.submit(process_one_file, path, file)
            )

        for f in as_completed(futures):
            # 如果子进程炸了，这里会抛异常
            f.result()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
if __name__ == "__main__":
    root = r"H:\Falcor\datasets\impostor"
    impostor_list = os.listdir(root)

    for impostor in impostor_list:
        path = os.path.join(root, impostor, "level1")
        if not os.path.isdir(path):
            continue

        print(f"[INFO] Processing {path}")
        process_impostor(path)

# impostor_list = os.listdir(r"H:\Falcor\datasets/impostor/")
# for impostor in impostor_list:
#     path = r"H:\Falcor\datasets\impostor/{}\level1".format(impostor)
#     file_list = os.listdir(path)
#     for file in file_list:
#         if not file.isdigit():
#             continue
#         dd = int(file)
#         formatted_dd = "{:05}".format(dd)  # 使用格式化字符串
#         image_path = path + r"/{}/depth_{}.exr".format(file,formatted_dd)
#         print(image_path)
#         data = pyexr.read(image_path)
#         data_tensor = torch.Tensor(data)
#         for level in range(10):
#             down_scale = 1<<level
#             value = downsample_depth_minmax(data_tensor.unsqueeze(0), down_scale, down_scale).squeeze(0).cpu().numpy()
#             target_image_path =  path + r"/{}/{}depth.exr".format(file,level)
#             print(down_scale)
#             pyexr.write(target_image_path, value)
#     for file in file_list:
#         if not file.isdigit():
#             continue
#         dd = int(file)
#         formatted_dd = "{:05}".format(dd)  # 使用格式化字符串
#         image_path = path + r"/{}/normal_{}.exr".format(file,formatted_dd)
#         print(image_path)
#         data = pyexr.read(image_path)

        # target_image_path =  path + r"/{}/normal.exr".format(file)
        # pyexr.write(target_image_path, data)

