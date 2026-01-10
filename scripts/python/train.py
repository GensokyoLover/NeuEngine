from impostor import *
import cv2
import numpy as np
from dataset import *
from utils import *
import os
from network import *
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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

W, H = 64, 64

move_speed = 0.02
rot_speed = 3 * math.pi / 180   # 3 degrees

i = 0


# dataset_process(r"H:\Falcor\datasets\renderdata\dragon_reftest/",impostor,test=False)
# exit()
#ckpt_path = r"H:\Falcor\ckpt\cylinder_encoding_lr_nearestgg3e-4_continue/step_00000090.pt" 
ckpt_path =False
start_epoch = 0
roughness = torch.round(torch.arange(0.0, 1.0 + 1e-9, 0.01) * 100) / 100
theta95_deg = ggx_theta_from_roughness(roughness, alpha_is_roughness_sq=True, degrees=True,energy_clamp=75).cuda()
# dataset_process(r"H:/Falcor/media/inv_rendering_scenes/bunny_ref_nobunny_roughnesscorrect/",impostor)
# exit()
impostor =load_zst(r"H:\Falcor\datasets\impostor\flame\level1/impostor.pkl.zst")
label = "roughness_9_3_cylinder_encoder2"
log_dir = "./runs/{}".format(label)
ensure_dir(log_dir)
writer = SummaryWriter(log_dir=log_dir)

debug_dir = "./output/{}".format(label)
ensure_dir(debug_dir)
ckpt_save_path = "./ckpt/{}".format(label)
ensure_dir(ckpt_save_path)
global_step = 0
training_data = ImpostorTrainingDataset(r"H:/Falcor/datasets/renderdata/flame/train")
#test_data = ImpostorTrainingTestDataset(r"H:/Falcor/media/inv_rendering_scenes/bunny_ref_nobunny9_25/")
training_data_loader = DataLoader(training_data, batch_size=1, shuffle=True)
#test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)
emissive_encoder = MultiScaleImageEncoder(3,128).cuda()
geo_encoder = MultiScaleImageEncoder(6,128).cuda()
image_decoder = ImageDecoder(10 + 128,256).cuda()
image_decoder2 = ImageDecoder(10,256).cuda()
mlp_weight = MLP15to3Softmax().cuda()


if ckpt_path != False:
    start_epoch, global_step = load_checkpoint(
        ckpt_path,
        emissive_encoder,
        geo_encoder,
        image_decoder,
        image_decoder2,
        mlp_weight,
        optimizer=None
    )
params = []
params += list(emissive_encoder.parameters())
params += list(geo_encoder.parameters())
params += list(image_decoder.parameters())
params += list(image_decoder2.parameters())
params += list(mlp_weight.parameters())

optimizer = torch.optim.AdamW(
    params,
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)
# view_combine = PixelMultiheadAttention(128).cuda()

criterion = torch.nn.L1Loss()   
num_epochs = 1000


# roughness: 0.00, 0.01, ..., 1.00


# 打印 CSV（roughness,theta95_deg）
# for r, th in zip(roughness, theta95_deg):
#     print(f"{r:.2f},{th:.6f}")
sample_key = r"reference_uvi2"
for epoch in range(num_epochs):
    print(f"===== Epoch {epoch+1}/{num_epochs} =====")

    # tqdm: 直接包住 dataloader 最方便（不需要你手动 iter/next）
    pbar = tqdm(training_data_loader, total=len(training_data_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True)
    total_loss = 0.0
    for i, (train_data, scene_id) in enumerate(pbar):
        train_data = tocuda(train_data)
        scale = 1.0/ impostor.cRadius[0]
        viewidx = train_data["sampleView"][:, 0, 0, :]
        train_data["position"] = train_data["position"] * scale
        for b in range(3):
            train_data["idepth{}".format(b)] = train_data["idepth{}".format(b)] * scale
        train_data["hitdepth"] = train_data["idepth0"][...,3:4] 
        depthDownList = []
        positionDownList = []
        emissionDownList = [[],[],[]]
        for b in range(viewidx.shape[0]):
            emission = sellect_data(impostor.texDict, viewidx[b].long(), ["emission"])
            depth = sellect_data(impostor.texDict, viewidx[b].long(), ["depth"])
            direction = sellect_data(impostor.texDict, viewidx[b].long(), ["view"])
            origin = sellect_data(impostor.texDict, viewidx[b].long(), ["raypos"])
            normal = sellect_data(impostor.texDict, viewidx[b].long(), ["normal"])
            centor = origin - direction * depth
        depthDownList.append(depth)
        positionDownList.append(origin)
        for b in range(3):
            emissionDownList[b].append(emission[b:b+1,...].permute(0,3,1,2))
            for level in range(4):
                depthDownList.append(min_downsample_pool(depth, 4**(level+1)))
                positionDownList.append(mean_downsample_pool(origin, 4**(level+1)))
                emissionDownList[b].append(mean_downsample_pool(emission[b:b+1,...], 4**(level+1)).permute(0,3,1,2))
        
        gbuffer_input = sellect_gbuffer_data(train_data, ["position", "normal", "roughness", "view"])
        feature_list = []
        #print(train_data["roughness"].mean())
        for b in range(3):
            emission_feature = emissive_encoder(emission[b:b+1,...])
            feature = trilinear_mipmap_sample(emission_feature,train_data["uv{}".format(b)])
            emission_vis = trilinear_mipmap_sample(emissionDownList[b],train_data["uv{}".format(b)])
            # pyexr.write(r"H:\Falcor\debug/emission_vis_{}.exr".format(b),emission_vis[0,...].permute(1,2,0).cpu().numpy())
            # pyexr.write(r"H:\Falcor\debug/gt_vis_{}.exr".format(b),train_data["AccumulatePassoutput"][0,...].cpu().numpy())
            feature_list.append(feature.permute(0,2,3,1))
        
        reflect_cone = torch.cat([train_data["position"],train_data["reflect"],train_data["roughness"],train_data["hitdepth"]],dim=-1)
        reflect_position = train_data["position"] + train_data["reflect"] * train_data["hitdepth"]
        for b in range(3):
            view_depth = ((reflect_position - impostor.cPosition[viewidx[0,b].long()].view(1,1,1,3)) * impostor.cForward[viewidx[0,b].long()].view(1,1,1,3)).sum(dim=-1,keepdim=True)
            normalize_depth = (train_data["idepth{}".format(b)][...,0:2] - view_depth) / (train_data["idepth{}".format(b)][...,2:3] + 1e-2)
            #reflect_cone = torch.cat([reflect_cone,train_data["idirection{}".format(b)][...,:3],train_data["idepth{}".format(b)][...,:2]],dim=-1)
            reflect_cone = torch.cat([reflect_cone,train_data["idirection{}".format(b)][...,:3],normalize_depth],dim=-1)
            # pyexr.write(r"H:\Falcor\debug/normalize_depth_{}.exr".format(b),normalize_depth[0,...].cpu().numpy())
            # pyexr.write(r"H:\Falcor\debug/train_data_{}.exr".format(b),train_data["idepth{}".format(b)][0,...].cpu().numpy())
            # pyexr.write(r"H:\Falcor\debug/view_depth_{}.exr".format(b),view_depth[0,...].cpu().numpy())
        #continue
        weights = mlp_weight(reflect_cone)
        feature = torch.cat(feature_list,dim=0)
        combine_feature = (feature * weights.permute(3, 1, 2, 0)).sum(dim=0, keepdim=True)
       

        decoder_input = torch.cat([combine_feature, gbuffer_input], dim=-1)
        #decoder_input = torch.cat([combine_feature, gbuffer_input], dim=-1)
        #output = image_decoder2(gbuffer_input)  # 假设输出 (1,H,W,3)
        output = image_decoder(decoder_input)  # 假设输出 (1,H,W,3)

        # -----------------------
        # loss + backward
        # -----------------------
        # gt：你注释里用的是 AccumulatePassoutput，按你实际 gt 对齐
        gt = train_data["AccumulatePassoutput"].reshape(1, H, W, 3)

        # 这里按需改 loss（L1/MSE/LPIPS等）
        loss = criterion(output, gt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total_loss = total_loss + loss.item()
        optimizer.step()

        # -----------------------
        # tensorboard logging
        # -----------------------
        

        # tqdm postfix
        pbar.set_postfix(loss=f"{loss.item():.6f}", step=global_step)
        if (global_step % 10) == 0 and i < 20:
            save_path = os.path.join(debug_dir, f"step_{global_step:08d}_scene_{i}.exr")
            save_result_exr(output.detach(), gt.detach(), save_path)
        #break
    writer.add_scalar("train/loss", float(total_loss/len(training_data_loader)), global_step)
    global_step += 1
    if global_step % 10 == 0:
        save_checkpoint(
            ckpt_save_path + f"/step_{global_step:08d}.pt",
            epoch,
            global_step,
            emissive_encoder,
            geo_encoder,
            image_decoder,
            image_decoder2,
            mlp_weight
        )
    continue
    #continue
    tpbar = tqdm(test_data_loader, total=len(test_data_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True)
    ## test no grad
    emissive_encoder.eval()
    mlp_weight.eval()
    image_decoder.eval()
    # 2. 禁用梯度（最重要）
    with torch.no_grad():
        debug_test_dir = debug_dir + "/test/"
        ensure_dir(debug_test_dir)
        for i, (test_data, scene_id) in enumerate(tpbar):
            if i != 4:
                continue
            test_data = tocuda(test_data)
            emission_feature_list = []
            for v in range(42):
                emission_feature = emissive_encoder(impostor.texDict["emission"][v:v+1])
                torch.cuda.empty_cache()
                print(torch.cuda.memory_allocated() / 1024**3, "GB allocated")
                print(torch.cuda.memory_reserved()  / 1024**3, "GB reserved")
                emission_feature_list.append(emission_feature["I0"])
            emission_feature = torch.cat(emission_feature_list,dim=0)
     
       
            final_feature = sample_by_uvi_bilinear_align_false(emission_feature.permute(0,2,3,1), test_data[sample_key].reshape(1,512,512,3,3),mode="nearest")

            depth = impostor.texDict["depth"]
            origin = impostor.texDict["raypos"]
            depthDownList = []
            positionDownList = []
            depthDownList.append(depth)
            positionDownList.append(origin)
            for level in range(4):
                depthDownList.append(min_downsample_pool(depth, 4**(level+1)))
                positionDownList.append(mean_downsample_pool(origin, 4**(level+1)))
            gbuffer_input = sellect_gbuffer_data(test_data, ["position", "normal", "roughness", "view"])
            #geo_feature = geo_encoder(geo_input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            

            # emission_vis = F.grid_sample(emission.permute(0, 3, 1, 2), reflect_uvi, mode="nearest", align_corners=False)
            emission_vis = sample_by_uvi_bilinear_align_false(impostor.texDict["emission"],test_data[sample_key].reshape(1,512,512,3,3),mode="nearest")
            referCylinderAxis = sample_by_uvi_bilinear_align_false(-impostor.texDict["view"], test_data[sample_key].reshape(1,512,512,3,3),mode="nearest")
            firstHit = sample_by_uvi_bilinear_align_false(depthDownList[1], test_data[sample_key].reshape(1,512,512,3,3),mode="nearest")
            referOrigin = sample_by_uvi_bilinear_align_false(origin, test_data[sample_key].reshape(1,512,512,3,3),mode="nearest")
            referCylinderRadius = (torch.zeros_like(firstHit) + 16/512)
            referCylinderLenth = referCylinderRadius * 2
            referCylinderCentor = (firstHit + referCylinderRadius) * referCylinderAxis + referOrigin
            referCylinderLenth = referCylinderLenth
            referCylinderCentor = referCylinderCentor
            referCylinderAxis = referCylinderAxis
            singleCentor = referCylinderCentor[0,:,:,:].permute(1,2,0)
            singleCentor1 = referCylinderCentor[1,:,:,:].permute(1,2,0)
            singleCentor2 = referCylinderCentor[2,:,:,:].permute(1,2,0)
            targetCylinderAxis = test_data["reflectDirL"].reshape(1,512,512,-1)
            targetConeAngle = theta95_deg[(torch.where(test_data["roughness"] * 100<2,2,test_data["roughness"] * 100)).int()]
            targetCylinderRadius = cone_radius_deg(test_data["mind"],targetConeAngle)
            targetCylinderLenth = targetCylinderRadius * 2
            targetCylinderCentor = test_data["referencePosL2"].reshape(1,512,512,-1)
            value = relative_cylinder_encoding_with_axis(referCylinderCentor,referCylinderAxis,referCylinderRadius[...,0],referCylinderLenth[...,0],targetCylinderCentor,targetCylinderAxis,targetCylinderRadius[...,0],targetCylinderLenth[...,0])
            reflect_input =value.permute(1,2,0,3).reshape(1,512,512,15)
            weight = mlp_weight(reflect_input)  # 期望 (1,256,256,3) or 等价

         
            combine_feature = (final_feature * weight.permute(3, 1, 2, 0)).sum(dim=0, keepdim=True)

            decoder_input = torch.cat([combine_feature, gbuffer_input], dim=-1)
            #decoder_input = torch.cat([combine_feature, gbuffer_input], dim=-1)
            #output = image_decoder2(gbuffer_input)  # 假设输出 (1,H,W,3)
            output = image_decoder(decoder_input)  # 假设输出 (1,H,W,3)
            output_path = os.path.join(debug_test_dir, f"epoch_{global_step+1:04d}_scene_{i}_output.exr")
            gt_path = os.path.join(debug_test_dir, f"epoch_{global_step+1:04d}_scene_{i}_gt.exr")
            roughness_path = os.path.join(debug_test_dir, f"epoch_{global_step+1:04d}_scene_{i}_roughness.exr")
            pyexr.write(gt_path,test_data["AccumulatePassoutput"].reshape(512,512,3).cpu().numpy())
            pyexr.write(output_path,output.reshape(512,512,3).cpu().numpy())
            pyexr.write(roughness_path,test_data["roughness"].reshape(512,512,1).cpu().numpy())
            break
    emissive_encoder.train()
    mlp_weight.train()
    image_decoder.train()
writer.close()
# for epoch in range(num_epochs):
#     print(f"===== Epoch {epoch+1}/{num_epochs} =====")
#     ## get dataset
#     data_iter = iter(training_data_loader)
   
#     lenth = len(training_data_loader)
#     for i in range(lenth):   # 遍历所有 datasetsp
#         train_data,scene_id = next(data_iter)
#         train_data = tocuda(train_data)
#         viewidx = train_data["reflect_uvi"][:,0,0,:,2]
#         for b in range(viewidx.shape[0]):
#             emission = sellect_data(impostor.texDict,viewidx[b].long(),["emission"])
#             depth = sellect_data(impostor.texDict,viewidx[b].long(),["depth"])
#             direction = sellect_data(impostor.texDict,viewidx[b].long(),["view"])
#             origin = sellect_data(impostor.texDict,viewidx[b].long(),["raypos"])
#             centor = origin - direction * depth
        
#         gbuffer_input = sellect_gbuffer_data(train_data,["position","normal","roughness","view"])
#         emission_feature = image_encoder(emission.permute(0,3,1,2)).permute(0,2,3,1)
#         reflect_uvi = train_data["reflect_uvi"][...,:2].permute(0,3,1,2,4).reshape(-1,H,W,2)
#         #reflect_uvi = reflect_uvi[...,[1,0]]
#         emission_vis = torch.nn.functional.grid_sample(emission.permute(0,3,1,2),reflect_uvi,mode='bilinear',align_corners=False)
#         out_dir = torch.nn.functional.grid_sample(-direction.permute(0,3,1,2),reflect_uvi,mode='bilinear',align_corners=False)
#         reflect_centor = torch.nn.functional.grid_sample(centor.permute(0,3,1,2),reflect_uvi,mode='bilinear',align_corners=False)
#         gbuffer_to_centor = reflect_centor.permute(0,2,3,1) - train_data["refelctPosL"] 
#         angle = angle_between_tensors_atan2(gbuffer_to_centor,train_data["reflectDirL"],dim=-1,degrees=True) / 0.1
#         reflect_lenth = gbuffer_to_centor.norm(dim=-1,keepdim=True) 
#         mini,_ = reflect_lenth.min(dim=0,keepdim=True)
#         reflect_lenth = reflect_lenth - mini

#         reflect_input = torch.cat([out_dir.permute(0,2,3,1),angle.unsqueeze(dim=-1),reflect_lenth],dim=-1)
#         reflect_input = reflect_input.permute(1,2,0,3).reshape(1,256,256,15)
#         weight = mlp_weight(reflect_input)
       
#         # for i in range(3):
#         #     pyexr.write("H:/Falcor/media/inv_rendering_scenes/object_level_config/Bunny/{}debug_emission{}.exr".format(scene_id,i),emission_vis[i].permute(1,2,0).reshape(H,W,3).cpu().numpy())
#         #     pyexr.write("H:/Falcor/media/inv_rendering_scenes/object_level_config/Bunny/{}gtemission{}.exr".format(scene_id,i),train_data["AccumulatePassoutput"].reshape(H,W,3).cpu().numpy())
#         #     pyexr.write("H:/Falcor/media/inv_rendering_scenes/object_level_config/Bunny/{}angle{}.exr".format(scene_id,i),angle[i].reshape(H,W,1).cpu().numpy())
#         # continue
#         feature = torch.nn.functional.grid_sample(emission_feature.permute(0,3,1,2),reflect_uvi,mode='bilinear',align_corners=True).permute(0,2,3,1)
#         combine_feature = (feature * weight.permute(3,1,2,0)).sum(dim=0,keepdim=True)
#         decoder_input = torch.cat([combine_feature,gbuffer_input],dim=-1)
#         output = image_decoder(decoder_input)

  