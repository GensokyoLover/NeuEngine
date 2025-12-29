from impostor import *
import cv2
import numpy as np
from dataset import *
from utils import *
from visualize import *
def show_image(title, img):
    img = img.cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    cv2.imshow(title, img)
    cv2.waitKey(1)

def rotate_vector(v, axis, angle):
    # v, axis: (3,)
    # angle: radians
    axis = axis / torch.norm(axis)
    v = v / torch.norm(v)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    return v * cos + torch.cross(axis, v) * sin + axis * torch.dot(axis, v) * (1 - cos)

def upscale_to_1024(img_tensor):
    # img_tensor shape: (H, W, 3)
    img = img_tensor.permute(2,0,1).unsqueeze(0)  # → (1,3,H,W)

    img_up = torch.nn.functional.interpolate(
        img, 
        size=(1024,1024),
        mode="bilinear",
        align_corners=False
    )

    img_up = img_up.squeeze(0).permute(1,2,0)  # back to (1024,1024,3)
    return img_up


def generate_rays(W, H, camera_pos, forward, up=torch.tensor([0,1,0], dtype=torch.float32),
                  fov=60, ortho=False, device="cuda",scale=1):
    """
    生成世界坐标射线（rayPosW, rayDirW）
    
    W,H: 分辨率
    camera_pos: (3,)
    forward: (3,)
    up: (3,)
    fov: 仅对透视投影视用
    ortho: 是否使用正交投影
    """

    camera_pos = camera_pos.to(device)
    forward = F.normalize(forward.to(device), dim=0)
    up = F.normalize(up.to(device), dim=0)

    # 建立右方向
    right = F.normalize(torch.cross(forward, up), dim=0)
    up = F.normalize(torch.cross(right, forward), dim=0)

    # 归一化屏幕坐标
    i = torch.linspace(-1, 1, W, device=device)
    j = torch.linspace(-1, 1, H, device=device)
    px, py = torch.meshgrid(i, -j, indexing="xy")  # y 反转以与图像一致

    if ortho:
        # ------------------------
        #      正交投影
        # ------------------------
        ray_dir = forward.view(1,1,3).expand(H,W,3)

        # 计算屏幕平面上的偏移
        scale = 1  # orthographic size，可改成参数
        offset = (px.unsqueeze(-1) * right + py.unsqueeze(-1) * up) * scale

        ray_pos = camera_pos.view(1,1,3) + offset

    else:
        # ------------------------
        #      透视投影
        # ------------------------
        fov_rad = (fov / 180.0) * 3.1415926
        z = 1 / torch.tan(fov_rad * 0.5)

        # 构造 NDC 的方向
        ndc = torch.stack((px, py, torch.full_like(px, -z)), dim=-1)

        # 旋转到世界坐标
        cam_mat = torch.stack([right, up, -forward], dim=1)   # 3x3
        ray_dir = torch.matmul(ndc.reshape(-1,3), cam_mat.T)
        ray_dir = F.normalize(ray_dir, dim=-1).reshape(H,W,3)

        # 所有透视射线起点相同
        ray_pos = camera_pos.view(1,1,3).expand(H,W,3)

    return ray_pos, ray_dir



    # 读取 depth.exr
# env CUDA_LAUNCH_BLOCKING=1n
training_datasets_list = ["AccumulatePassoutput","albedo","depth","specular","normal","position","roughness","view","raypos","emission"]
def read_training_datasets(idx):
    file_path = r"H:\Falcor\media\inv_rendering_scenes\bunny_ref_512_scale\{}/".format(idx)
    dt = {}
    for key in training_datasets_list:
        # 
        data = pyexr.read(file_path + key + "_{:05d}.exr".format(idx))[...,:buffer_channel[key]]
        dt[key] = torch.Tensor(data).cuda()
    with open(file_path + "node.json","r") as file:
        node = json.load(file)
    dt["node"] = node
    return dt
def sellect_data(texDict,idx,key_list):
    result = []
    for key in key_list:
        result.append(texDict[key][idx,:,:,:])
    return torch.cat(result,dim=-1)

def sellect_gbuffer_data(texDict,key_list):
    result = []
    for key in key_list:
        result.append(texDict[key])
    return torch.cat(result,dim=-1)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
## dataloader
from torch.utils.data import DataLoader
impostor = load_impostor("Light",1)

camera_pos = impostor.cPosition[0].clone().float().cuda()
forward    = impostor.cForward[0].clone().float().cuda()
up         = torch.tensor([0,1,0], dtype=torch.float32).cuda()

W, H = 256, 256

move_speed = 0.02
rot_speed = 3 * math.pi / 180   # 3 degrees

i = 0

roughness = torch.round(torch.arange(0.0, 1.0 + 1e-9, 0.01) * 100) / 100
theta95_deg = ggx_theta99_from_roughness(roughness, alpha_is_roughness_sq=True, degrees=True).cuda()
path = r"H:/Falcor/media/inv_rendering_scenes/bunny_ref_nobunny0/"
file_list = os.listdir(path)
impostor = load_impostor("Light",1)
for scene_str in file_list:
    if scene_str == "buckets_1":
        continue
    scene_id = int(scene_str)
    folder = os.path.join(path, f"{scene_id}")
    sample = {}

    for key in  ["AccumulatePassoutput","albedo","depth","specular","normal","position","roughness","view","raypos","emission"]:
        exr_path = os.path.join(folder, f"{key}_{scene_id:05d}.exr")

        data = pyexr.read(exr_path)[..., : buffer_channel[key]]

        sample[key] = torch.Tensor(data).cuda()  # shape: (H,W,C)
    sample["mask"] = (sample["position"].sum(dim=-1,keepdim=True)!=0) & (sample["roughness"] < 0.0001) & (sample["normal"][...,2:3]<0.1)
        
    if sample["mask"].sum() < 1:
        print("continue ",scene_id)
        continue
    json_path = os.path.join(folder, "node.json")
    with open(json_path, "r") as f:
        node = json.load(f)

    sample["node"] = node  # dict，不转 GPU
    train_data = sample
    W,H,_ =train_data["raypos"].shape
    
    rayPosW,rayDirW = train_data["raypos"].reshape(-1,3),-train_data["view"].reshape(-1,3)
    normal = train_data["normal"].reshape(-1,3)
    reflectPosW = train_data["position"].reshape(-1,3)
    reflectDirW = compute_reflection_direction(rayDirW,normal)
    
    print(torch.Tensor(train_data["node"]["Light"]["transform"]))
    rayPos_local, rayDir_local = transform_rays_world_to_local(
        rayPosW,
        rayDirW,
        torch.Tensor(train_data["node"]["Light"]["transform"])
    )
    refelctPosL, reflectDirL = transform_rays_world_to_local(
        reflectPosW,
        reflectDirW,
        torch.Tensor(train_data["node"]["Light"]["transform"])
    )
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    first_uvi,_ = chunk_process(impostor, rayDir_local, rayPos_local, W=W, H=H, chunk=512*512//8)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"[TIMER] chunk_process(first_uvi): {(t1 - t0)*1000:.3f} ms")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    half_angle_deg = theta95_deg[(train_data["roughness"].reshape(-1)*100).long()].reshape(-1)
    

    reflect_uvi,final_pos = chunk_process(impostor, reflectDirL, refelctPosL, W=W, H=H, chunk=512*512//8 )
    
    N = rayDir_local.shape[0]
    sample["reflect_uvi"] = reflect_uvi
    sample["first_uvi"] = reflect_uvi
    sample["refelctPosL"] = refelctPosL
    sample["reflectDirL"] = reflectDirL
    last = show_5_images_click_single_red_block(sample, impostor, block=8)
exit()
dataset_process(r"H:/Falcor/media/inv_rendering_scenes/bunny_ref_nobunny1/",impostor)
# exit()
# dataset_process(r"H:/Falcor/media/inv_rendering_scenes/bunny_ref_nobunny_roughnesscorrect/",impostor)
# exit()
impostor = load_impostor2("Light",1)
log_dir = "./runs/impostor_train2"
ensure_dir(log_dir)
writer = SummaryWriter(log_dir=log_dir)

debug_dir = "./debug_images"
ensure_dir(debug_dir)

global_step = 0
training_data = ImpostorTrainingDataset(r"H:/Falcor/media/inv_rendering_scenes/bunny_ref_nobunny_roughnesscorrect/")
training_data_loader = DataLoader(training_data, batch_size=1, shuffle=True)
image_encoder = TextureEncoderDilated_NoBN(3,64,128,16).cuda()
image_decoder = ImageDecoder(10 + 128,256).cuda()
mlp_weight = MLP15to3Softmax().cuda()
# view_combine = PixelMultiheadAttention(128).cuda()
params = []
params += list(image_encoder.parameters())
params += list(image_decoder.parameters())
params += list(mlp_weight.parameters())

optimizer = torch.optim.AdamW(
    params,
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)
criterion = torch.nn.L1Loss()   
num_epochs = 1000


# roughness: 0.00, 0.01, ..., 1.00


# 打印 CSV（roughness,theta95_deg）
for r, th in zip(roughness, theta95_deg):
    print(f"{r:.2f},{th:.6f}")
for epoch in range(num_epochs):
    print(f"===== Epoch {epoch+1}/{num_epochs} =====")

    # tqdm: 直接包住 dataloader 最方便（不需要你手动 iter/next）
    pbar = tqdm(training_data_loader, total=len(training_data_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True)
    total_loss = 0.0
    for i, (train_data, scene_id) in enumerate(pbar):
        train_data = tocuda(train_data)
        
#         viewidx = train_data["reflect_uvi"][:, 0, 0, :, 2]

#         # 注意：你这里每个 b 都在覆盖 emission/depth/... 变量
#         # 如果你希望 batch 内每个 b 各自有一套 emission/depth，需要把它们堆起来；否则这里只会保留最后一个 b 的结果
#         for b in range(viewidx.shape[0]):
#             emission = sellect_data(impostor.texDict, viewidx[b].long(), ["emission"])
#             depth = sellect_data(impostor.texDict, viewidx[b].long(), ["depth"])
#             direction = sellect_data(impostor.texDict, viewidx[b].long(), ["view"])
#             origin = sellect_data(impostor.texDict, viewidx[b].long(), ["raypos"])
#             centor = origin - direction * depth

#         gbuffer_input = sellect_gbuffer_data(train_data, ["position", "normal", "roughness", "view"])

#         emission_feature = image_encoder(emission.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

#         reflect_uvi = train_data["reflect_uvi"][..., :2].permute(0, 3, 1, 2, 4).reshape(-1, H, W, 2)

#         emission_vis = F.grid_sample(emission.permute(0, 3, 1, 2), reflect_uvi, mode="bilinear", align_corners=False)
#         out_dir = F.grid_sample(-direction.permute(0, 3, 1, 2), reflect_uvi, mode="bilinear", align_corners=False)
#         reflect_centor = F.grid_sample(centor.permute(0, 3, 1, 2), reflect_uvi, mode="bilinear", align_corners=False)

#         gbuffer_to_centor = reflect_centor.permute(0, 2, 3, 1) - train_data["refelctPosL"]
#         angle = angle_between_tensors_atan2(gbuffer_to_centor, train_data["reflectDirL"], dim=-1, degrees=True) / 0.1

#         reflect_lenth = gbuffer_to_centor.norm(dim=-1, keepdim=True)
#         mini, _ = reflect_lenth.min(dim=0, keepdim=True)
#         reflect_lenth = reflect_lenth - mini

#         reflect_input = torch.cat([out_dir.permute(0, 2, 3, 1), angle.unsqueeze(dim=-1), reflect_lenth], dim=-1)
#         reflect_input = reflect_input.permute(1, 2, 0, 3).reshape(1, 256, 256, 15)

#         weight = mlp_weight(reflect_input)  # 期望 (1,256,256,3) or 等价

#         feature = F.grid_sample(
#             emission_feature.permute(0, 3, 1, 2),
#             reflect_uvi,
#             mode="bilinear",
#             align_corners=True
#         ).permute(0, 2, 3, 1)  # (3,H,W,C) or (V,H,W,C)

#         # weight: (1,256,256,3) -> (3,256,256,1) to broadcast with feature
#         # 你原来是 weight.permute(3,1,2,0) => (3,256,256,1)
#         combine_feature = (feature * weight.permute(3, 1, 2, 0)).sum(dim=0, keepdim=True)

#         decoder_input = torch.cat([combine_feature, gbuffer_input], dim=-1)
#         output = image_decoder(decoder_input)  # 假设输出 (1,H,W,3)

#         # -----------------------
#         # loss + backward
#         # -----------------------
#         # gt：你注释里用的是 AccumulatePassoutput，按你实际 gt 对齐
#         gt = train_data["AccumulatePassoutput"].reshape(1, H, W, 3)

#         # 这里按需改 loss（L1/MSE/LPIPS等）
#         loss = criterion(output, gt)

#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         total_loss = total_loss + loss.item()
#         optimizer.step()

#         # -----------------------
#         # tensorboard logging
#         # -----------------------
        

#         # tqdm postfix
#         pbar.set_postfix(loss=f"{loss.item():.6f}", step=global_step)
#         if (global_step % 10) == 0 and i < 20:
#             save_path = os.path.join(debug_dir, f"step_{global_step:08d}_scene_{i}.exr")
#             save_result_exr(output.detach(), gt.detach(), save_path)

#     writer.add_scalar("train/loss", float(total_loss/len(training_data_loader)), global_step)
    

#     global_step += 1

#     # 如果你真要“每 1000 epoch 保存”，把上面的 if 换成下面这个：
#     # if ((epoch + 1) % 1000) == 0:  ... 保存一次

# # 训练结束关闭 writer
# writer.close()

  