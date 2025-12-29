import random

import numpy
import torch
from .shadow_network import *
from .partition_pyramid import *
from .transformer import TransformerDecoder
from .nrsm import *
from .utils import *
from einops import rearrange
from .unet import *
import gzip
import numpy as np
from .autoencoder import *
from torchvision.models.vision_transformer import VisionTransformer, ConvStemConfig, _log_api_usage_once, \
    Conv2dNormActivation, OrderedDict, Encoder, EncoderBlock, MLPBlock
# from .deepspeed import PositionwiseFeedforwardLayer
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from functools import partial
import pickle
from networks import modules
import torch
import torch.nn as nn
import time 
class Timer:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.start_ts = time.perf_counter()
    
    def end(self):
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event)  # ms
        else:
            return (time.perf_counter() - self.start_ts) * 1000.0 

def wrap_sample(img, uv, wrap_u=False, wrap_v=False):
    """
    img: (B, C, H, W)
    uv: (B, N, 2)
    wrap_u: 是否对 u wrap (表示 φ)
    wrap_v: 是否对 v wrap
    """

    B, C, H, W = img.shape
    u = uv[..., 0] * W - 0.5
    v = uv[..., 1] * H - 0.5

    # bilinear 4 neighbors
    u0 = torch.floor(u)
    v0 = torch.floor(v)
    u1 = u0 + 1
    v1 = v0 + 1

    fu = u - u0
    fv = v - v0

    # wrap or clamp per dimension
    def idx(val, size, do_wrap):
        if do_wrap:
            return (val.long() % size)
        else:
            return val.long().clamp(0, size - 1)

    u0i = idx(u0, W, wrap_u)
    u1i = idx(u1, W, wrap_u)
    v0i = idx(v0, H, wrap_v)
    v1i = idx(v1, H, wrap_v)

    # gather
    def gather(uu, vv):
        return img[..., vv, uu]  # (B,N,C)

    c00 = gather(u0i, v0i)
    c10 = gather(u1i, v0i)
    c01 = gather(u0i, v1i)
    c11 = gather(u1i, v1i)

    # bilinear weight
    c0 = c00 * (1 - fu)[...,None] + c10 * fu[...,None]
    c1 = c01 * (1 - fu)[...,None] + c11 * fu[...,None]
    out = c0 * (1 - fv)[...,None] + c1 * fv[...,None]
    return out

def octahedral_project(xyz):
    """
    xyz: (...,3)，必须是已归一化的方向向量
    return: uv in [-1,1]
    """
    x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]
    abs_sum = x.abs() + y.abs() + z.abs() + 1e-8

    # 初步投影
    u = x / abs_sum
    v = y / abs_sum

    # 下半球折叠
    mask = (z < 0)
    u2 = (1 - v.abs()) * u.sign()
    v2 = (1 - u.abs()) * v.sign()

    u = torch.where(mask, u2, u)
    v = torch.where(mask, v2, v)

    # 归一化到 [0,1]
    uv = torch.stack([u, v], dim=-1)
    return (uv + 1) * 0.5
def normalize(x):
    return x / torch.sqrt((x * x).sum(dim=-1)).unsqueeze(dim=-1)


def find_closest_intersection(positions, directions, radius=1):
    # 取批量射线位置和方向

    # 二次方程的系数 A, B, C，计算每条射线的系数
    A = torch.sum(directions ** 2, dim=-1)  # B, N
    B_coeff = 2 * torch.sum(positions * directions, dim=-1)  # B, N
    C = torch.sum(positions ** 2, dim=-1) - radius ** 2  # B, N

    # 计算判别式
    discriminant = B_coeff ** 2 - 4 * A * C  # B, N

    # 如果判别式小于0，表示没有交点，返回inf
    no_intersection = discriminant < 0

    # 计算t值：求解二次方程
    t1 = (-B_coeff + torch.sqrt(torch.clamp(discriminant, min=0))) / (2 * A)  # B, N
    t2 = (-B_coeff - torch.sqrt(torch.clamp(discriminant, min=0))) / (2 * A)  # B, N

    t_values = torch.stack([t1, t2], dim=-1)  # B, N, 2
    t_min = torch.min(t_values, dim=-1).values  # B, N
    # 对于没有交点的射线，返回inf
    t_min[no_intersection] = float('inf')
    # 计算交点
    intersections = positions + t_min.unsqueeze(-1) * directions  # B, N, 2
    return intersections


def rotate(a, b, c, v):
    return torch.cat([torch.sum((a * v), dim=-1).unsqueeze(-1),
                      torch.sum((b * v), dim=-1).unsqueeze(-1),
                      torch.sum((c * v), dim=-1).unsqueeze(-1)], dim=-1)


def compute_angle(u, v):
    # 计算点积
    dot_product = torch.sum(u * v, dim=-1)

    # 计算余弦值
    cos_theta = dot_product / (torch.norm(u, dim=-1) * torch.norm(v, dim=-1))

    # 通过反余弦计算夹角（单位：弧度）
    angle = torch.acos(torch.clamp(cos_theta, min=-1.0, max=1.0)).unsqueeze(-1)

    return angle


def get_lod_tracing(lposition, specular_ray, normal):
    lenth = torch.norm(lposition, keepdim=True, dim=-1)
    gg = torch.asin(1 / lenth)
    light_range = torch.cat([torch.pi / 2 - gg, torch.pi / 2 + gg], dim=-1)
    z = normalize(-lposition)
    normalize_ray = normalize(specular_ray)
    up = normalize(torch.cross(z, specular_ray, dim=-1))
    right = normalize(torch.cross(up, z, dim=-1))
    rotation_matrix = torch.cat([right.unsqueeze(-2), z.unsqueeze(-2), up.unsqueeze(-2)], dim=-2)
    local_ray = (rotation_matrix @ specular_ray.unsqueeze(-1)).squeeze(-1)
    relative_angular = torch.atan(local_ray[..., 1:2] / (local_ray[..., 0:1]))
    lod_angular = torch.Tensor([0 / 180, 15 / 180, 45 / 180, 90 / 180]) * torch.pi
    final_cast_ray = []
    local_lposition = (rotation_matrix @ lposition.unsqueeze(-1)).squeeze(-1)[..., :2]
    local_normal = (rotation_matrix @ normal.unsqueeze(-1)).squeeze(-1)
    result_mask = []
    # result_hit = []
    result_uv = []
    spherical_uv = []
    angular_range_list = []
    space_range_list = []
    local_point_left_list = []
    standard_dir_list = []
    local_point_right_list = []
    shuaijian_list = []
    # pyexr.write("../testData/relative_angular.exr",relative_angular[0].cpu().numpy())
    extend_angular = [15 / 180 * torch.pi, 30 / 180 * torch.pi, 45 / 180 * torch.pi, 0.01 / 180 * torch.pi]
    for i in range(4):

        ray_range = torch.cat([relative_angular - lod_angular[i], relative_angular + lod_angular[i]], dim=-1)
        left_min = torch.max(ray_range[..., :1], light_range[..., :1])
        right_max = torch.min(ray_range[..., 1:2], light_range[..., 1:2])
        left_cha = light_range[..., :1] - ray_range[..., 1:2]
        left_cha = torch.where(left_cha < 0, 0, left_cha)
        shuaijian = 1 - left_cha / extend_angular[i]
        shuaijian = torch.where(shuaijian < 0, 0, shuaijian)

        final_range = torch.cat([left_min, right_max], dim=-1)

        ray_mask = left_min <= (right_max + 1e-5)

        ray_left = torch.cat([torch.cos(left_min), torch.sin(left_min)], dim=-1)
        ray_right = torch.cat([torch.cos(right_max), torch.sin(right_max)], dim=-1)
        ray_mid = torch.cat([torch.cos((left_min + right_max) / 2), torch.sin((left_min + right_max) / 2)], dim=-1)
        point_left = find_closest_intersection(local_lposition, ray_left)
        left_angular = torch.atan(point_left[..., 1:2] / (point_left[..., 0:1] + 1e-5))
        left_angular = torch.where(left_min == light_range[..., :1], -gg, left_angular)
        point_left = torch.cos(left_angular) * right + torch.sin(left_angular) * z

        local_point_left = torch.cat([torch.cos(left_angular), torch.sin(left_angular)], dim=-1) * 1
        local_to_left = local_point_left - local_lposition
        point_left = point_left * 1
        to_left = normalize(point_left - lposition)
        point_right = find_closest_intersection(local_lposition, ray_right)

        right_angular = torch.atan(point_right[..., 1:2] / (point_right[..., 0:1] + 1e-5))
        right_angular = torch.where(right_max == light_range[..., 1:2], -torch.pi + gg, right_angular)
        right_angular = torch.where(right_angular > 0, right_angular - torch.pi, right_angular)
        point_right = torch.cos(right_angular) * right + torch.sin(right_angular) * z
        local_point_right = torch.cat([torch.cos(right_angular), torch.sin(right_angular)], dim=-1) * 1
        local_to_right = local_point_right - local_lposition

        point_right = point_right * 1
        to_right = normalize(point_right - lposition)
        angular_range = torch.abs(right_angular - left_angular)

        point_mid = find_closest_intersection(local_lposition, ray_mid)
        new_lenth = torch.sqrt(lenth * lenth - 1 * 1)
        ##pyexr.write("../testData/lenth{}.exr".format(i),new_lenth[0].cpu().numpy())
        standard_left_point_x = new_lenth * torch.cos(light_range[..., :1])
        standard_left_point_y = new_lenth * torch.sin(light_range[..., :1]) - lenth
        standard_point = standard_left_point_x * right + standard_left_point_y * z

        ##pyexr.write("../testData/standard_point{}.exr".format(i), standard_point[0].cpu().numpy())
        true_point = point_mid[..., :1] * right + point_mid[..., 1:2] * z
        true_point = torch.where(~ray_mask.repeat(1, 1, 1, 3), standard_point, true_point)
        to_standard_dir = normalize(true_point - lposition)
        standard_dir_list.append(to_standard_dir)
        normalize_pt = normalize(true_point)
        uv = EqualAreaSphereToSquare(normalize(normalize_pt))
        reflect_forward, reflect_up, reflect_right = get_forward_up_right_tensor(-normalize_pt)
        reflect_up = -reflect_up
        uvw = rotate(reflect_right, reflect_up, reflect_forward, to_standard_dir)
        standard_dir = normalize(uvw)
        another_uv = concentric_mapping_hemisphere_3D_to_2D(standard_dir)
        angular_range_list.append(angular_range)
        space_range = compute_angle(to_left, to_right)
        space_range_list.append(space_range)
        result_uv.append(uv)
        spherical_uv.append(another_uv)
        local_point_left_list.append(local_to_left)
        local_point_right_list.append(local_to_right)
        if i == 3:
            ray_mask = (~torch.isnan(uv)[..., :1]) & (left_min <= (right_max + 1e-5))
        else:
            ray_mask = (~torch.isnan(uv)[..., :1])
        ray_mask = (~torch.isnan(another_uv)[..., :1]) & ray_mask
        shuaijian[~ray_mask] = 0
        # pyexr.write("../testData/shuaijian{}.exr".format(i), (shuaijian[0]).cpu().numpy())
        # pyexr.write("../testData/raymask{}.exr".format(i), ray_mask[0].cpu().numpy())
        # pyexr.write("../testData/uv{}.exr".format(i), uv[0].cpu().numpy())
        # pyexr.write("../testData/another_uv{}.exr".format(i), another_uv[0].cpu().numpy())
        result_mask.append(ray_mask)

        shuaijian_list.append(shuaijian)
    result_mask = torch.cat(result_mask, dim=0)
    # result_hit = []
    result_uv = torch.cat(result_uv, dim=0)
    spherical_uv = torch.cat(spherical_uv, dim=0)
    angular_range_list = torch.cat(angular_range_list, dim=0)
    space_range_list = torch.cat(space_range_list, dim=0)
    local_point_left_list = torch.cat(local_point_left_list, dim=0)
    shuaijian_list = torch.cat(shuaijian_list, dim=0)
    local_point_right_list = torch.cat(local_point_right_list, dim=0)
    standard_dir_list = torch.cat(standard_dir_list, dim=0)
    data = {}
    data["angular_uv"] = result_uv
    data["space_uv"] = spherical_uv
    data["angular_range"] = angular_range_list
    data["space_range"] = space_range_list
    data["local_point_left"] = local_point_left_list
    data["local_point_right"] = local_point_right_list
    data["true_point"] = local_point_right_list
    data["local_normal"] = local_normal
    data["local_ray"] = local_ray
    data["ray_mask"] = result_mask
    data["shuaijian"] = shuaijian_list
    data["standard_dir"] = standard_dir_list
    return data


def get_lod_tracing_arbitrary(lposition, specular_ray, normal, roughness, table, light_angular_size, light_space_size):
    data = {}
    _, W1, H1, _ = roughness.shape
    lenth = torch.norm(lposition, keepdim=True, dim=-1)
    gg = torch.asin(1 / lenth)
    light_range = torch.cat([torch.pi / 2 - gg, torch.pi / 2 + gg], dim=-1)
    z = normalize(-lposition)
    normalize_ray = normalize(specular_ray)
    up = normalize(torch.cross(z, specular_ray, dim=-1))
    right = normalize(torch.cross(up, z, dim=-1))
    rotation_matrix = torch.cat([right.unsqueeze(-2), z.unsqueeze(-2), up.unsqueeze(-2)], dim=-2)
    local_ray = (rotation_matrix @ specular_ray.unsqueeze(-1)).squeeze(-1)
    relative_angular = torch.atan(local_ray[..., 1:2] / (local_ray[..., 0:1]))
    data["x"] = torch.sin(relative_angular)
    data["y"] = (roughness * 2 - 1)
    data["z"] = ((lenth - 0.4) / 6) * 2 - 1
    roughness = (roughness * 1000).long()
    roughness = torch.clamp(roughness, min=0, max=999)
    # table = torch.Tensor(table).cuda()
    lod_angular = table[roughness.reshape(-1)].reshape(1, W1, H1, 1) * 2  # halfvec to reflectdir double 2

    ####print_image_exr(lod_angular,"lod_angular")
    ####print_image_exr(roughness,"roughness")
    final_cast_ray = []
    local_lposition = (rotation_matrix @ lposition.unsqueeze(-1)).squeeze(-1)[..., :2]
    local_normal = (rotation_matrix @ normal.unsqueeze(-1)).squeeze(-1)
    result_mask = []
    # result_hit = []
    result_uv = []
    spherical_uv = []
    angular_range_list = []
    space_range_list = []
    local_point_left_list = []
    standard_dir_list = []
    local_point_right_list = []
    ###print_image_exr(lod_angular,"lod_angular")
    ray_range = torch.cat([relative_angular - lod_angular, relative_angular + lod_angular], dim=-1)
    ###print_image_exr(ray_range[...,:1], "ray_range0")
    ###print_image_exr(ray_range[...,1:2], "ray_range1")
    clamp_ray_range = torch.clamp(ray_range, min=0, max=torch.pi)
    ray_mid_after_clamp = (clamp_ray_range[..., :1] + clamp_ray_range[..., 1:2]) / 2

    left_min = torch.max(ray_range[..., :1], light_range[..., :1])
    right_max = torch.min(ray_range[..., 1:2], light_range[..., 1:2])
    # compute the scale when cone is nearly outside the 4d representation
    scale = (lod_angular * 2 / (right_max - left_min + 1e-4)) * (lod_angular * 2 / (right_max - left_min + 1e-4))

    scale2, cone2, cone_bin = cone_union_percentage_radians(torch.clamp(gg * 1.3, min=0, max=torch.pi // 2),
                                                            torch.clamp(lod_angular * 1.3, min=0, max=torch.pi // 2),
                                                            torch.abs(ray_mid_after_clamp - torch.pi / 2))
    scale2 = torch.clamp(scale2, min=1)
    # ###print_image_exr(scale2,"scale2")
    # ###print_image_exr(cone2,"cone2")
    # ###print_image_exr(cone_bin,"cone_bin")
    # ###print_image_exr(gg,"gg")
    # ###print_image_exr(lod_angular,"lod_angular")
    # ###print_image_exr(torch.abs(ray_mid_after_clamp-torch.pi/2),"phi")

    final_range = torch.cat([left_min, right_max], dim=-1)

    ray_mask = left_min <= (right_max + 1e-6)

    ray_left = torch.cat([torch.cos(left_min), torch.sin(left_min)], dim=-1)
    ray_right = torch.cat([torch.cos(right_max), torch.sin(right_max)], dim=-1)
    ###print_image_exr(ray_mid_after_clamp, "ray_mid_after_clamp")
    ###print_image_exr((left_min + right_max) / 2, "ray_mid")

    ray_mid = torch.cat([torch.cos((left_min + right_max) / 2), torch.sin((left_min + right_max) / 2)], dim=-1)

    point_left = find_closest_intersection(local_lposition, ray_left)
    ####print_image_exr(ray_range[...,:1],"ray_range0")
    ####print_image_exr(ray_range[...,1:2],"ray_range1")
    ####print_image_exr(light_range[...,:1],"light_range0")
    ####print_image_exr(light_range[...,1:2],"light_range1")
    ####print_image_exr(left_min,"left_min")
    ####print_image_exr(right_max,"right_max")
    ####print_image_exr(relative_angular,"relative_angular")
    left_angular = torch.atan(point_left[..., 1:2] / (point_left[..., 0:1] + 1e-5))
    left_angular = torch.where(left_min == light_range[..., :1], -gg, left_angular)

    point_left = torch.cos(left_angular) * right + torch.sin(left_angular) * z
    ####print_image_exr(point_left, "point_left")
    local_point_left = torch.cat([torch.cos(left_angular), torch.sin(left_angular)], dim=-1) * 1
    local_to_left = local_point_left - local_lposition
    point_left = point_left * 1
    to_left = normalize(point_left - lposition)
    point_right = find_closest_intersection(local_lposition, ray_right)

    right_angular = torch.atan(point_right[..., 1:2] / (point_right[..., 0:1] + 1e-5))
    right_angular = torch.where(right_angular > 0, right_angular - torch.pi, right_angular)
    right_angular = torch.where(right_max == light_range[..., 1:2], - torch.pi + gg, right_angular)
    point_right = torch.cos(right_angular) * right + torch.sin(right_angular) * z
    angular_lenth = torch.abs(right_angular - left_angular) / torch.pi / 2
    space_lenth = torch.abs(right_max - left_min) / torch.pi
    ####print_image_exr(point_right,"point_right")
    local_point_right = torch.cat([torch.cos(right_angular), torch.sin(right_angular)], dim=-1) * 1
    local_to_right = local_point_right - local_lposition

    point_right = point_right * 1

    to_right = normalize(point_right - lposition)
    angular_range = torch.abs(right_angular - left_angular)

    point_mid = find_closest_intersection(local_lposition, ray_mid)
    ####print_image_exr(point_mid,"point_mid")
    new_lenth = torch.sqrt(lenth * lenth - 1 * 1)
    ##pyexr.write("../testData/lenth{}.exr".format(i),new_lenth[0].cpu().numpy())
    standard_left_point_x = new_lenth * torch.cos(light_range[..., :1])
    standard_left_point_y = new_lenth * torch.sin(light_range[..., :1]) - lenth
    standard_point = standard_left_point_x * right + standard_left_point_y * z

    ##pyexr.write("../testData/standard_point{}.exr".format(i), standard_point[0].cpu().numpy())
    true_point = point_mid[..., :1] * right + point_mid[..., 1:2] * z
    to_standard_dir = normalize(true_point - lposition)
    to_left_dir = normalize(point_left - lposition)
    to_right_dir = normalize(point_right - lposition)

    standard_dir_list.append(to_standard_dir)
    normalize_pt = normalize(true_point)
    uv = EqualAreaSphereToSquare(normalize(normalize_pt))
    left_uv = EqualAreaSphereToSquare(normalize(point_left))
    right_uv = EqualAreaSphereToSquare(normalize(point_right))
    reflect_forward1, reflect_up1, reflect_right1 = get_forward_up_right_tensor(-normalize_pt)
    reflect_up1 = -reflect_up1

    reflect_right = torch.cat([reflect_right1[..., :1], reflect_up1[..., :1], reflect_forward1[..., :1]], dim=-1)
    reflect_up = torch.cat([reflect_right1[..., 1:2], reflect_up1[..., 1:2], reflect_forward1[..., 1:2]], dim=-1)
    reflect_forward = torch.cat([reflect_right1[..., 2:3], reflect_up1[..., 2:3], reflect_forward1[..., 2:3]], dim=-1)

    # uvw = rotate(reflect_right, reflect_up, reflect_forward, to_standard_dir)
    # uvw_left = rotate(reflect_right, reflect_up, reflect_forward, to_left_dir)
    # uvw_right = rotate(reflect_right, reflect_up, reflect_forward, to_right_dir)

    uvw = rotate(reflect_right1, reflect_up1, reflect_forward1, to_standard_dir)
    uvw_left = rotate(reflect_right1, reflect_up1, reflect_forward1, to_left_dir)
    uvw_right = rotate(reflect_right1, reflect_up1, reflect_forward1, to_right_dir)
    standard_dir = normalize(uvw)
    left_dir = normalize(uvw_left)
    right_dir = normalize(uvw_right)
    another_uv = concentric_mapping_hemisphere_3D_to_2D(standard_dir)
    left_space_uv = concentric_mapping_hemisphere_3D_to_2D(left_dir)
    right_space_uv = concentric_mapping_hemisphere_3D_to_2D(right_dir)
    angular_range_list.append(angular_range)
    space_range = compute_angle(to_left, to_right)
    space_range_list.append(space_range)
    result_uv.append(uv)
    spherical_uv.append(another_uv)
    local_point_left_list.append(local_to_left)
    local_point_right_list.append(local_to_right)

    ray_mask = (~torch.isnan(uv)[..., :1]) & ray_mask
    ray_mask = (~torch.isnan(another_uv)[..., :1]) & ray_mask

    result_mask.append(ray_mask)

    result_mask = torch.cat(result_mask, dim=0)
    ####print_image_exr(ray_mask,"ray_mask")
    # ###print_image_exr(uv,"angular_uv")
    # ###print_image_exr(another_uv,"space_uv")
    # ###print_image_exr(ray_mask,"ray_mask")
    # ####print_image_exr(left_uv,"left_uv")
    ####print_image_exr(right_uv,"right_uv")
    # result_hit = []
    result_uv = torch.cat(result_uv, dim=0)
    spherical_uv = torch.cat(spherical_uv, dim=0)
    angular_range_list = torch.cat(angular_range_list, dim=0)
    space_range_list = torch.cat(space_range_list, dim=0)
    local_point_left_list = torch.cat(local_point_left_list, dim=0)
    local_point_right_list = torch.cat(local_point_right_list, dim=0)
    standard_dir_list = torch.cat(standard_dir_list, dim=0)

    data["angular_uv"] = result_uv
    data["angular_uv_lenth"] = angular_lenth
    data["right_angular_uv"] = right_uv

    data["space_uv"] = spherical_uv
    data["space_uv_lenth"] = space_lenth
    ###print_image_exr(data["space_uv_lenth"],"space_uv_lenth")
    ###print_image_exr(data["angular_uv_lenth"],"angular_uv_lenth")
    data["space_uv_left"] = torch.clamp((data["space_uv"] - data["space_uv_lenth"] + 1) / 2 * light_space_size // 1,
                                        min=0, max=light_space_size)
    data["space_uv_right"] = torch.clamp(
        (data["space_uv"] + data["space_uv_lenth"] + 1) / 2 * light_space_size // 1 + 1, min=0, max=light_space_size)
    data["angular_uv_left"] = torch.clamp(
        (data["angular_uv"] - data["angular_uv_lenth"] + 1) / 2 * light_angular_size // 1, min=0,
        max=light_angular_size)
    data["angular_uv_right"] = torch.clamp(
        (data["angular_uv"] + data["angular_uv_lenth"] + 1) / 2 * light_angular_size // 1 + 1, min=0,
        max=light_angular_size)

    data["space_uv_left"] = data["space_uv_left"][..., [1, 0]]
    data["space_uv_right"] = data["space_uv_right"][..., [1, 0]]
    data["angular_uv_left"] = data["angular_uv_left"][..., [1, 0]]
    data["angular_uv_right"] = data["angular_uv_right"][..., [1, 0]]

    data["space_uv_left1"] = (data["space_uv"] - data["space_uv_lenth"] + 1) / 2 * light_space_size
    data["space_uv_right1"] = (data["space_uv"] + data["space_uv_lenth"] + 1) / 2 * light_space_size + 1
    data["angular_uv_left1"] = (data["angular_uv"] - data["angular_uv_lenth"] + 1) / 2 * light_angular_size
    data["angular_uv_right1"] = (data["angular_uv"] + data["angular_uv_lenth"] + 1) / 2 * light_angular_size + 1
    ###print_image_exr(data["space_uv_left1"],"space_uv_left1")
    ###print_image_exr(data["space_uv_right1"],"space_uv_right1")
    ###print_image_exr(data["angular_uv_left1"],"angular_uv_left1")
    ###print_image_exr(data["angular_uv_right1"],"angular_uv_right1")

    data["angular_range"] = angular_range_list
    data["space_range"] = space_range_list
    data["local_point_left"] = local_point_left_list
    data["local_point_right"] = local_point_right_list
    data["true_point"] = local_point_right_list
    data["local_normal"] = local_normal
    data["local_ray"] = local_ray
    data["ray_mask"] = result_mask
    data["standard_dir"] = standard_dir_list
    data["cone_scale"] = scale
    data["cone_scale2"] = scale2
    return data


def get_light_space_gbuffer(data,forward_indirect):
    lposition = data["local"]["gbuffer"]["lposition"]
    lenth = torch.norm(lposition, keepdim=True, dim=-1)
    z = normalize(-lposition)
    reflect_forward1, reflect_up1, reflect_right1 = get_forward_up_right_tensor(z)

    localGbuffer = {}
    localGbuffer["specular_ray"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                          data["local"]["gbuffer"]["specular_ray"])
    localGbuffer["normal"] = rotate(reflect_right1, reflect_up1, reflect_forward1, data["local"]["gbuffer"]["normal"])
    localGbuffer["view_dir"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                      data["local"]["gbuffer"]["view_dir"])
    localGbuffer["half_vec"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                      data["local"]["gbuffer"]["half_vec"])
    if forward_indirect:
        light_position = data["local"]["lights"]["shadow"]["light_position"]
        lenth = torch.norm(light_position, keepdim=True, dim=-1)
        z = normalize(-light_position)
        reflect_forward1, reflect_up1, reflect_right1 = get_forward_up_right_tensor(z)
        data["local"]["lights"]["shadow"]["light_lnormal"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                                                data["local"]["lights"]["shadow"]["light_normal"])
        data["local"]["lights"]["shadow"]["light_half_vec"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                                                data["local"]["lights"]["shadow"]["light_half_vec"])
        data["local"]["lights"]["shadow"]["light_specular_ray"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                                                data["local"]["lights"]["shadow"]["light_specular_ray"])
    return localGbuffer


def oct_transformer2(intuv, size):
    result_int = intuv
    bool_a = (intuv < 0)
    bool_b = (intuv >= size)
    result_int[bool_a] = 0
    result_int[bool_b] = size - 1
    bool_uv = bool_a | bool_b
    bool_v = bool_uv[..., 1].clone()
    bool_uv[..., 1] = bool_uv[..., 0]
    bool_uv[..., 0] = bool_v

    result_int = torch.where(bool_uv, size - 1 - result_int, result_int)
    result_uv = (result_int + 0.5) / size * 2 - 1
    return result_uv


def oct_transform(uv, size):
    B = uv.shape[:-1]
    C = uv.shape[-1:]
    float_uv = (uv + 1) / 2 * size
    intuv = ((uv + 1) / 2 * size).long()
    cha = float_uv - intuv
    intuv_ll = torch.where(cha < 0.5, intuv - 1, intuv)
    cha = torch.where(cha < 0.5, cha + 0.5, cha - 0.5)
    biasu = torch.zeros_like(intuv).cuda()
    biasu[..., :1] = 1
    biasv = torch.zeros_like(intuv).cuda()
    biasv[..., 1:2] = 1
    intuv_rl = intuv_ll + biasu
    intuv_lr = intuv_ll + biasv
    intuv_rr = intuv_ll + biasu + biasv
    uv_ll_mid = oct_transformer2(intuv_ll, size)
    uv_rl_mid = oct_transformer2(intuv_rl, size)
    uv_lr_mid = oct_transformer2(intuv_lr, size)
    uv_rr_mid = oct_transformer2(intuv_rr, size)
    a = (1 - cha[..., 0:1]) * (1 - cha[..., 1:2])
    b = (cha[..., 0:1]) * (1 - cha[..., 1:2])
    c = (1 - cha[..., 0:1]) * (cha[..., 1:2])
    d = (cha[..., 0:1]) * (cha[..., 1:2])

    weight = torch.cat([a, b, c, d], dim=-1)
    return uv_ll_mid, uv_rl_mid, uv_lr_mid, uv_rr_mid, weight


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def active_func(act):
    if act == "sigmoid":
        return torch.sigmoid()


class BasicTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ff_dim=2048, dropout=0.1):
        """
        dim: 输入通道数 (c)
        num_heads: 注意力头数
        ff_dim: 前馈层隐藏维度
        dropout: dropout比例
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (n, l, c)
        """
        # --- Self-Attention ---
        attn_out, _ = self.attn(x, x, x)           # 自注意力
        x = self.norm1(x + self.dropout(attn_out)) # 残差连接 + LN

        # --- Feed Forward ---
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))   # 残差连接 + LN
        return x

class DitBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            mlp_ratio=4.0,
            moe_config=None,
            soft_max=True
    ):
        super().__init__()
        self.num_heads = num_heads
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # Attention block
        if norm_layer:
            self.ln_1 = norm_layer(hidden_dim)
            self.ln_2 = norm_layer(hidden_dim)
            print("layer norm yes")
        else:
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()
            #exit()
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        # MLP block
        if moe_config == None:
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            print("lets create moe")
            # self.mlp = PositionwiseFeedforwardLayer(hidden_dim, moe_config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    # def forward(self, x: torch.Tensor,c):
    #     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
    #     prex = x
    #     sa = modulate(self.ln_1(x),shift_msa,scale_msa)
    #     sa,_ = self.self_attention(sa,sa,sa,need_weights=False)
    #     x = prex +  gate_msa.unsqueeze(dim=1) * sa
    #     x = x + gate_mlp.unsqueeze(dim=1) * self.mlp(modulate(self.ln_2(x),shift_mlp,scale_mlp))
    #     return x
    def forward(self, x: torch.Tensor, c=None):
        prex = x
        # #print(self.ln_1.type)
        sa = self.ln_1(x)
        # print(sa.shape)f
        sa, _ = self.self_attention(sa, sa, sa, need_weights=False, average_attn_weights=False)
        x = prex + sa
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = True,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cond_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
        )
        self.query_layer = nn.Linear(inner_dim, cond_dim)
        self.value_layer = nn.Linear(cond_dim, cond_dim)
        self.key_layer = nn.Linear(cond_dim, cond_dim)

    def forward(self, x, cond):

        light, weight = self.cross_attn(self.query_layer(x), self.key_layer(cond), self.value_layer(cond),
                                        need_weights=False)
        return light



class TriCrossAttentionBlock(nn.Module):
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int,
                 shared_attn: bool = False
                ):
        super().__init__()

        self.shared_attn = shared_attn

        # === attention模块 ===
        if shared_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=inner_dim, num_heads=num_heads,
                kdim=cond_dim, vdim=cond_dim,
                batch_first=True
            )
        else:
            self.cross_attn_xy = nn.MultiheadAttention(
                embed_dim=inner_dim, num_heads=num_heads,
                kdim=cond_dim, vdim=cond_dim,
                batch_first=True)
            self.cross_attn_xz = nn.MultiheadAttention(
                embed_dim=inner_dim, num_heads=num_heads,
                kdim=cond_dim, vdim=cond_dim,
                batch_first=True)
            self.cross_attn_yz = nn.MultiheadAttention(
                embed_dim=inner_dim, num_heads=num_heads,
                kdim=cond_dim, vdim=cond_dim,
                batch_first=True)

        # === Q/K/V 投影层 ===
        self.xy_key_layer = nn.Linear(inner_dim, inner_dim)
        self.xz_key_layer = nn.Linear(inner_dim, inner_dim)
        self.yz_key_layer = nn.Linear(inner_dim, inner_dim)

        self.xy_query_layer = nn.Linear(cond_dim, cond_dim)
        self.xz_query_layer = nn.Linear(cond_dim, cond_dim)
        self.yz_query_layer = nn.Linear(cond_dim, cond_dim)

        self.xy_value_layer = nn.Linear(cond_dim, cond_dim)
        self.xz_value_layer = nn.Linear(cond_dim, cond_dim)
        self.yz_value_layer = nn.Linear(cond_dim, cond_dim)


    def forward(self, x, cond):
        """
        x:    [B, 3*N, D]   triplane features (flattened)
        cond: [B, L, D_cond]  light field tokens
        """
        b = x.shape[0]
        x = x.reshape(b, 3, -1, x.shape[-1])
        xy_key = x[:, 0, ...]
        xz_key = x[:, 1, ...]
        yz_key = x[:, 2, ...]

        cond_pe = cond

        if self.shared_attn:

            Q_xy = self.xy_query_layer(xy_key)
            Q_xz = self.xz_query_layer(xz_key)
            Q_yz = self.yz_query_layer(yz_key)

            K = self.xy_key_layer(cond_pe)   # 这里可以共享 K/V 映射
            V = self.xy_value_layer(cond_pe)

            xy_light, _ = self.cross_attn(xy_key, K, V, need_weights=False)
            xz_light, _ = self.cross_attn(xz_key, K, V, need_weights=False)
            yz_light, _ = self.cross_attn(yz_key, K, V, need_weights=False)
            exit()
        else:
            # 每个平面独立 cross-attn
            xy_light, _ = self.cross_attn_xy(
                query=self.xy_query_layer(xy_key),
                key=self.xy_key_layer(cond_pe),
                value=self.xy_value_layer(cond_pe),
                need_weights=False
            )
            xz_light, _ = self.cross_attn_xz(
                query=self.xz_query_layer(xz_key),
                key=self.xz_key_layer(cond_pe),
                value=self.xz_value_layer(cond_pe),
                need_weights=False
            )
            yz_light, _ = self.cross_attn_yz(
                query=self.yz_query_layer(yz_key),
                key=self.yz_key_layer(cond_pe),
                value=self.yz_value_layer(cond_pe),
                need_weights=False
            )
        # === 合并 ===
        light = torch.cat([xy_light, xz_light, yz_light], dim=1)

        return light



class ConditionBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """

    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = True,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            batch_first=True)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
        )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        # weight = self.cross_attn(x, cond, cond, need_weights=True)[1]
        # #print("weight shape",weight.shape)
        # weight = torch.sum(weight,dim=-1)
        # #print(weight)
        x = x + self.cross_attn(x, cond, cond, need_weights=False)[0]
        x = x + self.self_attn(x, x, x, need_weights=False)[0]
        x = x + self.mlp(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlockTrue(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self,
                 num_heads: int,
                 hidden_size: int,
                 mlp_dim: int,
                 dropout: float,
                 attention_dropout: float,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 mlp_ratio=4.0,
                 moe_config=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.bn = Batch_Norm(hidden_size)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Batch_Norm(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.BN = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.BN(x)
        x = rearrange(x, 'b d n -> b n d')
        return x


class DitEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            plane_cnt: int = 8,
            moe_config=None,
            use_dit=False
    ):
        super().__init__()
        self.profile = False
        self.use_dit = use_dit
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(plane_cnt, seq_length, hidden_dim * 2).normal_(std=0.02))  # from BERT
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        if use_dit == False:
            for i in range(num_layers):
                layers[f"encoder_layer_{i}"] = DitBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                    4.0,
                    moe_config
                )
        else:
            for i in range(num_layers):
                layers[f"encoder_layer_{i}"] = DiTBlockTrue(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                    4.0,
                    moe_config
                )
        self.layers = nn.Sequential(layers)

    def forward(self, input: torch.Tensor, c: torch.Tensor, idx=0):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        mean_pos_embedding = self.pos_embedding.mean(dim=1).mean(dim=1)

        scale, bias = self.pos_embedding[idx].unsqueeze(dim=0).chunk(2, dim=-1)
        # #print(idx)
        input = input + bias
        cnt = 0
        for layer in self.layers:
            # #print(c.shape)
            input = layer(input, c)
            # #print(cnt,idx)
            cnt = cnt + 1
        return input



class DitTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            z_dim: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 1000,
            representation_size: Optional[int] = None,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            conv_stem_configs: Optional[List[ConvStemConfig]] = None,
            plane_cnt=8,
            config=None,
            use_dit=False
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=z_dim, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2
        moe_config = None
        self.encoder = DitEncoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
            plane_cnt,
            moe_config,
            use_dit=use_dit
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor, radius_embed_input, index):
        # Reshape and permute the input tensor
        n, c, h, w = x.shape
        x = self._process_input(x)
        n = x.shape[0]
        # #print("transformer shape : ",x.shape)
        x = self.encoder(x, radius_embed_input, index)
        # print("transformer shape : ", x.shape)
        x = x.permute(0, 2, 1).reshape(n, self.hidden_dim, h, w).permute(0, 2, 3, 1)
        return x


class FourDTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
            self,
            angular_size: int,
            space_size: int,
            angular_block_size: int,
            space_block_size: int,
            num_layers: int,
            num_heads: int,
            input_dim,
            hidden_dim: int,
            mlp_dim: int,
            output_dim: int,
            linear_output: bool,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.space_block_size = space_block_size
        self.angular_block_size = angular_block_size
        self.angular_size = angular_size
        self.space_size = space_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        input_dim = input_dim * self.space_block_size * self.space_block_size * self.angular_block_size * self.angular_block_size
        self.input_layer = nn.Linear(
            input_dim, hidden_dim
        )
        if linear_output:
            self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        else:
            self.output_layer = fc_layer(self.hidden_dim, output_dim)
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers * 2):
            layers[f"encoder_layer_{i}"] = DitBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                0.0,
                0.0,
                norm_layer,
                4.0
            )

        self.layers = nn.Sequential(layers)

    def forward(self, input: torch.Tensor):
        # Reshape and permute the input tensor
        B, W1, H1, W2, H2, C = input.shape
        # print(input.shape)
        input = input.reshape(B, W1 // self.angular_block_size, self.angular_block_size, H1 // self.angular_block_size,
                              self.angular_block_size, W2 // self.space_block_size, self.space_block_size,
                              H2 // self.space_block_size, self.space_block_size,
                              C).permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9).reshape(B, W1 // self.angular_block_size,
                                                                               H1 // self.angular_block_size,
                                                                               W2 // self.space_block_size,
                                                                               H2 // self.space_block_size, -1)
        # print("input shape",input.shape)

        input = self.input_layer(input)
        cnt = 0
        #print(input.shape)
        #exit()
        B, W1, H1, W2, H2, C = input.shape
        for layer in self.layers:
            # #print(c.shape)
            if cnt % 2 == 0:
                input = input.reshape(B * W1 * H1, W2 * H2, C)
            else:
                input = input.permute(0, 3, 4, 1, 2, 5).reshape(B * W2 * H2, W1 * H1, C)
            input = layer(input, None)
            if cnt % 2 == 0:
                input = input.reshape(B, W1, H1, W2, H2, C)
            else:
                input = input.reshape(B, W2, H2, W1, H1, C).permute(0, 3, 4, 1, 2, 5)
            # #print(cnt,idx)
            cnt = cnt + 1
            # print(input.shape)
        # print("transformer shape : ", x.shape)
        input = self.output_layer(input)
        return input



class FourDTransformerSimple(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
            self,
            angular_size: int,
            space_size: int,
            angular_block_size: int,
            space_block_size: int,
            num_layers: int,
            num_heads: int,
            input_dim,
            hidden_dim: int,
            mlp_dim: int,
            output_dim: int,
            linear_output: bool,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.space_block_size = space_block_size
        self.angular_block_size = angular_block_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        input_dim = input_dim * self.space_block_size * self.space_block_size * self.angular_block_size * self.angular_block_size
        self.input_layer = nn.Linear(
            input_dim, hidden_dim
        )
        if linear_output:
            self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        else:
            self.output_layer = fc_layer(self.hidden_dim, output_dim)
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers * 2):
            layers[f"encoder_layer_{i}"] = DitBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                0.0,
                0.0,
                norm_layer,
                4.0,
                soft_max=False
            )

        self.layers = nn.Sequential(layers)

    def forward(self, input: torch.Tensor):
        # Reshape and permute the input tensor
        B, W1, H1, W2, H2, C = input.shape
        # print(input.shape)
        input = input.reshape(B, W1 // self.angular_block_size, self.angular_block_size, H1 // self.angular_block_size,
                              self.angular_block_size, W2 // self.space_block_size, self.space_block_size,
                              H2 // self.space_block_size, self.space_block_size,
                              C).permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9).reshape(B, W1 // self.angular_block_size,
                                                                               H1 // self.angular_block_size,
                                                                               W2 // self.space_block_size,
                                                                               H2 // self.space_block_size, -1)

        input = self.input_layer(input)
        input = input 
        cnt = 0
        B, W1, H1, W2, H2, C = input.shape
        for layer in self.layers:
            # #print(c.shape)
            if cnt % 2 == 0:
                input = input.reshape(B * W1 * H1, W2 * H2, C)
            else:
                input = input.permute(0, 3, 4, 1, 2, 5).reshape(B * W2 * H2, W1 * H1, C)
            input = layer(input, None)
            if cnt % 2 == 0:
                input = input.reshape(B, W1, H1, W2, H2, C)
            else:
                input = input.reshape(B, W2, H2, W1, H1, C).permute(0, 3, 4, 1, 2, 5)
            # #print(cnt,idx)
            cnt = cnt + 1
            # print(input.shape)
        # print("transformer shape : ", x.shape)
        input = self.output_layer(input)
        return input


class TransformerPlane(nn.Module):
    def __init__(self, configs, bounding_box=None, embedding_size=48, init_scale=1e-2):
        super().__init__()
        self.embedding_size = configs["direct_dim"]
        self.transformer = LightTransformer(image_size=16, patch_size=1, num_layers=6, num_heads=8, z_dim=48,
                                            hidden_dim=self.embedding_size,
                                            mlp_dim=self.embedding_size, representation_size=self.embedding_size)
        self.radius = 1
        self.plane_cnt = 10
        self.min_distance = self.radius
        self.distance_range = self.radius * (self.plane_cnt - 1)
        self.single_plane = False
        self.unetList = nn.ModuleList(
            [
                UNet(self.embedding_size, self.embedding_size
                     ) for i in range(1)])

    def forward_plane(self, data, photon_texture):
        plane_list = []
        B, W, H, C = photon_texture.shape
        photon_texture = photon_texture.permute(0, 3, 1, 2)
        self.voxel_grid = self.transformer(photon_texture)
        return

    def forward_grid(self, data, photon_texture):
        plane_list = []
        B, W, H, C = photon_texture.shape
        photon_texture = self.transformer(photon_texture.permute(0, 3, 1, 2))
        index = torch.zeros(B, 10, 64, 64, 1)

        for i in range(10):
            index[:, i, ...] = i
        for i in range(self.plane_cnt):
            if i == 0:
                plane_list.append(photon_texture)
            else:
                plane = plane_list[i - 1]
                plane = self.unetList[0](plane.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                plane_list.append(plane)
        for i in range(self.plane_cnt):
            # plane_list[i] = add_texture_border(plane_list[i])
            plane_list[i] = plane_list[i].unsqueeze(dim=1)
        self.voxel_grid = torch.cat(plane_list, dim=1)
        return

    def fetch_grid(self, dir):
        shape_len = len(dir.shape)
        if shape_len == 4:
            B, W, H, C = dir.shape
            N = W * H
            dir = dir.reshape(B, N, C)
        toLight = dir
        B, N, C = toLight.shape
        toLight = toLight.reshape(-1, N, C)
        toLightSquareDistance = torch.sum((toLight * toLight), dim=-1)
        toLightDistance = torch.sqrt(toLightSquareDistance)
        voxel_grid = self.voxel_grid
        B, D, W2, H2, C = voxel_grid.shape
        voxel_d = ((toLightDistance - self.min_distance) / self.distance_range).unsqueeze(dim=-1)
        toLightNormalization = toLight / toLightDistance.unsqueeze(dim=-1)
        gg = torch.sum(toLightNormalization * toLightNormalization, dim=-1)
        uv = EqualAreaSphereToSquare(toLightNormalization)  ## for
        u = uv[..., 1].clone()
        uv[..., 1] = uv[..., 0]
        uv[..., 0] = u
        voxel_d = voxel_d * 2 - 1
        voxel_coord = torch.cat([uv, voxel_d], dim=-1)
        voxel_grid = voxel_grid.reshape(B, D, W2, H2, C).permute(0, 4, 1, 2, 3)
        voxel_coord = voxel_coord.unsqueeze(dim=1).unsqueeze(dim=1)
        voxel_feature = torch.nn.functional.grid_sample(voxel_grid, voxel_coord)
        voxel_feature = voxel_feature.reshape(B, -1, N).permute(0, 2, 1)
        if shape_len == 4:
            voxel_feature = voxel_feature.reshape(B, W, H, -1)
        return voxel_feature

    def fetch_plane(self, dir):
        shape_len = len(dir.shape)
        if shape_len == 4:
            B, W, H, C = dir.shape
            N = W * H
            dir = dir.reshape(B, N, C)
        toLight = dir
        B, N, C = toLight.shape
        toLight = toLight.reshape(-1, N, C)
        toLightSquareDistance = torch.sum((toLight * toLight), dim=-1)
        toLightDistance = torch.sqrt(toLightSquareDistance)
        voxel_grid = self.voxel_grid
        B, W2, H2, C = voxel_grid.shape
        voxel_d = ((toLightDistance - self.min_distance) / self.distance_range).unsqueeze(dim=-1)
        toLightNormalization = toLight / toLightDistance.unsqueeze(dim=-1)
        gg = torch.sum(toLightNormalization * toLightNormalization, dim=-1)
        uv = EqualAreaSphereToSquare(toLightNormalization)  ## for
        u = uv[..., 1].clone()
        uv[..., 1] = uv[..., 0]
        uv[..., 0] = u
        voxel_coord = uv.unsqueeze(dim=1)
        voxel_grid = voxel_grid.reshape(B, W2, H2, C).permute(0, 3, 1, 2)
        voxel_feature = torch.nn.functional.grid_sample(voxel_grid, voxel_coord)
        voxel_feature = voxel_feature.reshape(B, -1, N).permute(0, 2, 1)
        if shape_len == 4:
            voxel_feature = voxel_feature.reshape(B, W, H, -1)
        return voxel_feature

class DiTransformerPlane(nn.Module):
    def __init__(self, configs, bounding_box=None, embedding_size=48, init_scale=1e-2):
        super().__init__()
        self.embedding_size = configs["encoder_direct_dim"]
        self.plane_config = configs["plane"]
        self.transport = configs["plane"]["transport"]
        if self.transport != "triplane_s":
            self.radius = 0.4
            self.ratio = self.plane_config["ratio"]
            self.plane_cnt = configs["plane"]["resolution"]
            self.max_radius = 7.67773699
        else:
            self.radius = 0.4
            self.max_radius = 7.67773699
            self.plane_cnt = configs["plane"]["resolution"]
            self.ratio = math.exp(math.log(self.max_radius / self.radius) / self.plane_cnt)
            # print(self.radius)
            # print(self.max_radius)

        self.up_scale = self.plane_config["up_scale"]
        self.min_distance = self.radius
        self.distance_range = self.max_radius - self.min_distance

        self.single_plane = False
        self.transformer_layer_cnt = configs["plane"]["transformer_cnt"]
        self.encoder_transformer_cnt = 6
        if "encoder_transformer_cnt" in configs["plane"].keys():
            self.encoder_transformer_cnt = configs["plane"]["encoder_transformer_cnt"]

        self.up_encoder = nn.ModuleList()

        embedding_size = configs["decoder_direct_dim"]
        for i in range(self.up_scale):
            self.up_encoder.append(nn.Sequential(nn.Conv2d(embedding_size, embedding_size, 3, padding='same'), \
                                                 nn.Conv2d(embedding_size, embedding_size, 3, padding='same'), \
                                                 nn.LeakyReLU(),
                                                 nn.Conv2d(embedding_size, max(embedding_size // 2, 32), 1),
                                                 nn.LeakyReLU()))
            embedding_size = max(embedding_size // 2, 32)
        if configs["plane"]["transport"] == "transformer" or configs["plane"]["transport"] == "residual":
            self.transformer = DitTransformer(image_size=32, patch_size=1, num_layers=self.transformer_layer_cnt,
                                              num_heads=16, z_dim=self.embedding_size,
                                              hidden_dim=self.embedding_size,
                                              mlp_dim=self.embedding_size,
                                              representation_size=self.embedding_size, plane_cnt=self.plane_cnt,
                                              config=configs, use_dit=self.plane_config["use_dit"], norm_layer=None)
        if configs["plane"]["transport"] == "standard":
            self.transformer = StandardTransformer(image_size=32, patch_size=1, num_layers=self.transformer_layer_cnt,
                                                   num_heads=16, z_dim=self.embedding_size,
                                                   hidden_dim=self.embedding_size,
                                                   mlp_dim=self.embedding_size,
                                                   representation_size=self.embedding_size, plane_cnt=self.plane_cnt,
                                                   config=configs, use_dit=self.plane_config["use_dit"],
                                                   norm_layer=None)
        elif configs["plane"]["transport"] == "unet":
            self.transformer = UNet(self.embedding_size, self.embedding_size)
        elif configs["plane"]["transport"] == "triplane" or configs["plane"]["transport"] == "triplane_s":
            self.transformer = DitTransformer(image_size=32, patch_size=1, num_layers=self.encoder_transformer_cnt,
                                              num_heads=16, z_dim=self.embedding_size,
                                              hidden_dim=self.embedding_size,
                                              mlp_dim=self.embedding_size,
                                              representation_size=self.embedding_size, plane_cnt=self.plane_cnt,
                                              config=configs, use_dit=self.plane_config["use_dit"], norm_layer=None)
            self.tri_plane_embed = nn.Parameter(0.01 * torch.randn(3, self.plane_cnt, self.plane_cnt, 256),
                                                requires_grad=True)
            self.transformer_decoder = CrossAttentionBlock(inner_dim=self.embedding_size, cond_dim=self.embedding_size,
                                                           num_heads=16, eps=1e-6)
        self.depth_fusion_layer = nn.Sequential(
            fc_layer(self.embedding_size * 2, self.embedding_size * 2),
            fc_layer(self.embedding_size * 2, self.embedding_size * 2),
            fc_layer(self.embedding_size * 2, self.embedding_size)
        )
        self.aux_depth_fusion_layer = nn.Sequential(
            fc_layer(self.embedding_size * 2, self.embedding_size * 2),
            fc_layer(self.embedding_size * 2, self.embedding_size * 2),
            fc_layer(self.embedding_size * 2, self.embedding_size)
        )
        self.idx = 0
        self.plane_list = []
        self.radius_embedding = nn.Parameter(
            torch.empty(self.plane_cnt, self.embedding_size).normal_(std=0.02))
        self.aux_radius_embed = nn.Parameter(
            torch.empty(self.plane_cnt, self.embedding_size).normal_(std=0.02))  # from BERT
        self.output_layer = nn.Linear(configs["encoder_direct_dim"], configs["decoder_direct_dim"])
        self.decoder_dim = configs["decoder_direct_dim"]
        self.radiance_linear_layer_list = nn.ModuleList()
        self.aux_linear_layer_list = nn.ModuleList()
        for i in range(9):
            self.radiance_linear_layer_list.append(nn.Linear(self.embedding_size, self.embedding_size))

        for i in range(9):
            self.aux_linear_layer_list.append(nn.Linear(self.embedding_size, self.embedding_size))

    def depth_fusion(self, plane, depth_embed):
        x = torch.cat([plane, depth_embed], dim=-1)
        return self.depth_fusion_layer(x)

    def aux_depth_fusion(self, plane, depth_embed):
        x = torch.cat([plane, depth_embed], dim=-1)
        return self.aux_depth_fusion_layer(x)

    def forward_plane(self, data, photon_texture):
        plane_list = []
        B, W, H, C = photon_texture.shape
        photon_texture = photon_texture.permute(0, 3, 1, 2)
        self.voxel_grid = self.transformer(photon_texture)
        return

    def up_sample(self, photon_texture):
        for layer in self.up_encoder:
            photon_texture = F.interpolate(photon_texture, scale_factor=2)
            photon_texture = layer(photon_texture)
            photon_texture = photon_texture
        return photon_texture

    def forward_new_triplane(self, data, photon_texture, aux_texture):
        query = self.tri_plane_embed.reshape(-1, self.embedding_size).unsqueeze(0).repeat(3, 1, 1)
        B, W, H, C = photon_texture.shape
        plane = self.transformer(photon_texture.permute(0, 3, 1, 2), photon_texture.permute(0, 3, 1, 2), 0)

        photon_texture = photon_texture.reshape(B, -1, C)
        triplane_feature = self.transformer_decoder(query, photon_texture)
        # #print(triplane_feature.shape)
        triplane_feature = triplane_feature.reshape(B, 3, self.plane_cnt, self.plane_cnt, self.embedding_size)
        triplane_feature = self.output_layer(triplane_feature)
        triplane_feature = self.up_sample(
            triplane_feature.reshape(9, self.plane_cnt, self.plane_cnt, self.decoder_dim).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1).reshape(3, 3, self.plane_cnt * (2 ** self.up_scale), self.plane_cnt * (2 ** self.up_scale), -1)
        # print("triplane: ", triplane_feature.shape)
        self.xy_feature = triplane_feature[:, 0, ...].squeeze(1).permute(0, 3, 1, 2)
        self.xz_feature = triplane_feature[:, 1, ...].squeeze(1).permute(0, 3, 1, 2)
        self.yz_feature = triplane_feature[:, 2, ...].squeeze(1).permute(0, 3, 1, 2)

    def forward_triplane(self, data, photon_texture, aux_texture):
        query = self.tri_plane_embed.reshape(-1, self.embedding_size).unsqueeze(0).repeat(3, 1, 1)
        B, W, H, C = photon_texture.shape
        plane = self.transformer(photon_texture.permute(0, 3, 1, 2), photon_texture.permute(0, 3, 1, 2), 0)

        photon_texture = plane.reshape(B, -1, C)
        triplane_feature = self.transformer_decoder(query, photon_texture)
        # #print(triplane_feature.shape)
        triplane_feature = triplane_feature.reshape(B, 3, self.plane_cnt, self.plane_cnt, self.embedding_size)
        triplane_feature = self.output_layer(triplane_feature)
        triplane_feature = self.up_sample(
            triplane_feature.reshape(9, self.plane_cnt, self.plane_cnt, self.decoder_dim).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1).reshape(3, 3, self.plane_cnt * (2 ** self.up_scale), self.plane_cnt * (2 ** self.up_scale), -1)
        # print("triplane: ", triplane_feature.shape)
        self.xy_feature = triplane_feature[:, 0, ...].squeeze(1).permute(0, 3, 1, 2)
        self.xz_feature = triplane_feature[:, 1, ...].squeeze(1).permute(0, 3, 1, 2)
        self.yz_feature = triplane_feature[:, 2, ...].squeeze(1).permute(0, 3, 1, 2)

    def forward_no_residual(self, data, photon_texture, aux_texture):
        B, W, H, C = photon_texture.shape
        # #print("photon shape ",photon_texture.shape)
        plane_list = []
        for i in range(self.plane_cnt):
            depth_embed = self.radius_embedding[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            aux_depth_embed = self.aux_radius_embed[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            depth_embed = depth_embed.repeat(B, W, H, 1)
            aux_depth_embed = aux_depth_embed.repeat(B, W, H, 1)
            plane_input = self.depth_fusion(photon_texture, depth_embed)
            aux_input = self.aux_depth_fusion(aux_texture, aux_depth_embed)
            # plane_input = self.radiance_linear_layer_list[i](photon_texture)
            # aux_input = self.aux_linear_layer_list[i](aux_texture)
            # #print("plane input : ",plane_input.shape)

            plane = self.transformer(plane_input.permute(0, 3, 1, 2), aux_input.permute(0, 3, 1, 2), i).permute(0, 3, 1,
                                                                                                                2)
            # #print("plane input:",plane.shape)
            plane = self.output_layer(plane.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            plane = self.output_layer(plane.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            plane = self.up_sample((plane)).permute(0, 2, 3, 1)
            # plane = torch.rand_like(plane)
            plane_list.append(plane)
        # #print("transformer photon size : ",plane_list[-1].shape)

        for i in range(self.plane_cnt):
            plane_list[i] = plane_list[i].unsqueeze(dim=1)
        self.voxel_grid = torch.cat(plane_list, dim=1)
        return

    def sample_d(self, r):
        if self.plane_config["sample_type"] == "log":
            k = torch.log(r / self.radius) / math.log(self.ratio)
            k_int = torch.floor(k)
            is_r_less_than_r0 = r < self.radius
            is_r_lager_than_rmax = r > self.max_radius
            radius_small = self.radius * torch.pow(self.ratio, k_int)
            radius_large = self.radius * torch.pow(self.ratio, k_int + 1)
            a = 1 / (self.plane_cnt - 1)
            # #print(r.shape)
            # #print(self.max_radius)
            
            #normalize_r = ((r - radius_small) / (radius_large - radius_small)).reshape(512, 512, 1)
            # ####pyexr.write("../testData/normalized_r_{}.exr".format(self.idx),numpy.array(normalize_r.detach().cpu()))

            linear_r = (r - radius_small) / (radius_large - radius_small) * a + k_int * a
            linear_r[is_r_less_than_r0] = 0
            linear_r[is_r_lager_than_rmax] = 1
            # ####pyexr.write("../testData/linear_{}.exr".format(self.idx), numpy.array(r.reshape(512,512,1).detach().cpu()))
            self.idx = self.idx + 1

        else:
            linear_r = (r - self.radius) / (self.max_radius - self.radius)
        return linear_r.unsqueeze(dim=-1)

    def fetch_tri_plane(self, dir):
        toLight = dir / 8
        xy = torch.cat([toLight[..., :1], toLight[..., 1:2]], dim=-1)
        xz = torch.cat([toLight[..., :1], toLight[..., 2:3]], dim=-1)
        yz = torch.cat([toLight[..., 1:2], toLight[..., 2:3]], dim=-1)
        xy_feature = torch.nn.functional.grid_sample(self.xy_feature, xy.repeat(3, 1, 1, 1)).permute(0, 2, 3, 1)
        xz_feature = torch.nn.functional.grid_sample(self.xz_feature, xz.repeat(3, 1, 1, 1)).permute(0, 2, 3, 1)
        yz_feature = torch.nn.functional.grid_sample(self.yz_feature, yz.repeat(3, 1, 1, 1)).permute(0, 2, 3, 1)
        # #print("xy_feature shape : ",xy_feature.shape)
        self.xy_feature = None
        self.xz_feature = None
        self.yz_feature = None
        return xy_feature + xz_feature + yz_feature, None



    def fetch_plane(self, dir):
        shape_len = len(dir.shape)
        if shape_len == 4:
            B, W, H, C = dir.shape
            N = W * H
            dir = dir.reshape(B, N, C)
        toLight = dir
        B, N, C = toLight.shape
        toLight = toLight.reshape(-1, N, C)
        toLightSquareDistance = torch.sum((toLight * toLight), dim=-1)
        toLightDistance = torch.sqrt(toLightSquareDistance)
        voxel_grid = self.voxel_grid
        B, W2, H2, C = voxel_grid.shape
        voxel_d = ((toLightDistance - self.min_distance) / self.distance_range).unsqueeze(dim=-1)
        toLightNormalization = toLight / toLightDistance.unsqueeze(dim=-1)
        gg = torch.sum(toLightNormalization * toLightNormalization, dim=-1)
        uv = EqualAreaSphereToSquare(toLightNormalization)  ## for
        u = uv[..., 1].clone()
        uv[..., 1] = uv[..., 0]
        uv[..., 0] = u
        voxel_coord = uv.unsqueeze(dim=1)
        voxel_grid = voxel_grid.reshape(B, W2, H2, C).permute(0, 3, 1, 2)
        voxel_feature = torch.nn.functional.grid_sample(voxel_grid, voxel_coord)
        voxel_feature = voxel_feature.reshape(B, -1, N).permute(0, 2, 1)
        if shape_len == 4:
            voxel_feature = voxel_feature.reshape(B, W, H, -1)
        return voxel_feature

class TriLinear(nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.xy_linear = nn.Linear(input_dim, output_dim)
        self.xz_linear = nn.Linear(input_dim, output_dim)
        self.yz_linear = nn.Linear(input_dim, output_dim)
    def forward(self,x):
        xy = x[:,0:1,...]
        xz = x[:,1:2,...]
        yz = x[:,2:3,...]
        xy = self.xy_linear(xy)
        xz = self.xz_linear(xz)
        yz = self.yz_linear(yz)
        return torch.cat([xy,xz,yz],dim=1)

class TriPosEncoder(nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.xy_linear = nn.Sequential(
                fc_layer(input_dim,output_dim),
                fc_layer(output_dim,output_dim),
                nn.Linear(output_dim,output_dim)
            )
        self.xz_linear = nn.Sequential(
            fc_layer(input_dim,output_dim),
            fc_layer(output_dim,output_dim),
            nn.Linear(output_dim,output_dim)
        )
        self.yz_linear = nn.Sequential(
            fc_layer(input_dim,output_dim),
            fc_layer(output_dim,output_dim),
            nn.Linear(output_dim,output_dim)
        )
    
    def forward(self,x):
        xy = x[0:1,...]
        xz = x[1:2,...]
        yz = x[2:3,...]
        xy = self.xy_linear(xy)
        xz = self.xz_linear(xz)
        yz = self.yz_linear(yz)
        return torch.cat([xy,xz,yz],dim=0)
    
def generate_triplane_coords(H=128, W=128, device='cuda'):
    """
    生成三张 2D 坐标 map: XY, YZ, XZ
    每个 map 的范围是 [-1, 1]
    """
    # 生成一维坐标
    lin = torch.linspace(-1, 1, H, device=device)
    x, y, z = torch.meshgrid(lin, lin, lin, indexing='ij')

    # 三张平面坐标
    xy = torch.stack([x[..., H//2], y[..., H//2]], dim=-1)  # 固定 z = 0 平面
    yz = torch.stack([y[H//2, ...], z[H//2, ...]], dim=-1)  # 固定 x = 0
    xz = torch.stack([x[:, H//2, :], z[:, H//2, :]], dim=-1)  # 固定 y = 0

    # [H, W, 2]
    return xy


class ChannelCutDiTransformerPlane(nn.Module):
    def __init__(self, configs, bounding_box=None, embedding_size=48, init_scale=1e-2,soft_max=False):
        super().__init__()
        self.embedding_size = configs["plane"]["encoder_direct_dim"]
        self.plane_config = configs["plane"]
        self.transport = configs["plane"]["transport"]
        if self.transport != "triplane_s":
            self.radius = 0.1
            self.ratio = self.plane_config["ratio"]
            self.plane_cnt = configs["plane"]["resolution"]
            self.max_radius = 7.67773699
        else:
            self.radius = 0.1
            self.max_radius = 7.67773699
            self.plane_cnt = configs["plane"]["resolution"]
            self.ratio = math.exp(math.log(self.max_radius / self.radius) / self.plane_cnt)
            # print(self.radius)
            # print(self.max_radius)

        self.up_scale = self.plane_config["up_scale"]
        self.min_distance = self.radius
        self.distance_range = self.max_radius - self.min_distance

        self.transformer_layer_cnt = configs["plane"]["transformer_cnt"]
        self.encoder_transformer_cnt = 6
        if "encoder_transformer_cnt" in configs["plane"].keys():
            self.encoder_transformer_cnt = configs["plane"]["encoder_transformer_cnt"]

        self.up_encoder = nn.ModuleList()


        
        
        self.decoder_dim = configs["plane"]["decoder_direct_dim"]
        self.radiance_linear_layer_list = nn.ModuleList()
        self.aux_linear_layer_list = nn.ModuleList()
        

        self.encoder_tri = configs["encoder_tri"]
        self.decoder_tri = configs["decoder_tri"]
        self.compress_tri = configs["compress_tri"]
        grid_map = generate_triplane_coords(self.plane_cnt,self.plane_cnt)
        self.pos_init = nn.Parameter(grid_map.unsqueeze(dim=0).repeat(3,1,1,1), requires_grad=False)
        if self.encoder_tri ==True:
            
            self.tri_encoder = TriPosEncoder(2,configs["plane"]["encoder_direct_dim"])
        else:
            self.tri_plane_embed = nn.Parameter(0.01 * torch.randn(3, self.plane_cnt, self.plane_cnt, configs["plane"]["encoder_direct_dim"]),
                                            requires_grad=True)
        
        if self.decoder_tri == True:
            self.tri_transformer_decoder = TriCrossAttentionBlock(inner_dim=self.embedding_size, cond_dim=self.embedding_size,
                                                           num_heads=16,shared_attn=False)
        else:
            self.transformer_decoder = CrossAttentionBlock(inner_dim=self.embedding_size, cond_dim=self.embedding_size,
                                                       num_heads=16, eps=1e-6)
            
        if self.compress_tri ==True:
            self.tri_output_layer = TriLinear(configs["plane"]["encoder_direct_dim"], configs["plane"]["decoder_direct_dim"])
        else:
            self.output_layer = nn.Linear(configs["plane"]["encoder_direct_dim"], configs["plane"]["decoder_direct_dim"])
    def forward_triplane(self, photon_texture):
        B,_,_,_,_,C = photon_texture.shape
        if self.encoder_tri == False:
            query = self.tri_plane_embed.reshape(-1, self.embedding_size).unsqueeze(0).repeat(B, 1, 1)
        else:
            query = self.tri_encoder(self.pos_init).reshape(-1,self.embedding_size).unsqueeze(0).repeat(B,1,1)
        
        photon_texture = photon_texture.reshape(B, -1, C)
        # print(photon_texture.shape)
        # print(query.shape)
        if self.decoder_tri == False:
            triplane_feature = self.transformer_decoder(query, photon_texture)
        else:
            triplane_feature = self.tri_transformer_decoder(query, photon_texture)
        # #print(triplane_feature.shape)
        triplane_feature = triplane_feature.reshape(B, 3, self.plane_cnt, self.plane_cnt, self.embedding_size)
        if self.compress_tri== False:
            compressed_triplane_feature = self.output_layer(triplane_feature)
        else:
            compressed_triplane_feature = self.tri_output_layer(triplane_feature)
        
        return triplane_feature,compressed_triplane_feature
    def sample_d(self, r):
        B,W,H = r.shape
        if self.plane_config["sample_type"] == "log":
            k = torch.log(r / self.radius) / math.log(self.ratio)
            k_int = torch.floor(k)
            is_r_less_than_r0 = r < self.radius
            is_r_lager_than_rmax = r > self.max_radius
            radius_small = self.radius * torch.pow(self.ratio, k_int)
            radius_large = self.radius * torch.pow(self.ratio, k_int + 1)
            a = 1 / (self.plane_cnt - 1)
            # #print(r.shape)
            # #print(self.max_radius)
            normalize_r = ((r - radius_small) / (radius_large - radius_small)).reshape(B, W, H)
            # ####pyexr.write("../testData/normalized_r_{}.exr".format(self.idx),numpy.array(normalize_r.detach().cpu()))

            linear_r = (r - radius_small) / (radius_large - radius_small) * a + k_int * a
            linear_r[is_r_less_than_r0] = 0
            linear_r[is_r_lager_than_rmax] = 1
            # ####pyexr.write("../testData/linear_{}.exr".format(self.idx), numpy.array(r.reshape(512,512,1).detach().cpu()))

        else:
            linear_r = (r - self.radius) / (self.max_radius - self.radius)
        return linear_r.unsqueeze(dim=-1)

    def fetch_tri_plane(self,texture, dir):
        toLight = dir / 6.4
        xy = torch.cat([toLight[..., :1], toLight[..., 1:2]], dim=-1)
        xz = torch.cat([toLight[..., :1], toLight[..., 2:3]], dim=-1)
        yz = torch.cat([toLight[..., 1:2], toLight[..., 2:3]], dim=-1)
        xy_feature = torch.nn.functional.grid_sample(texture[:,1,...].permute(0,3,1,2), xy).permute(0, 2, 3, 1)
        xz_feature = torch.nn.functional.grid_sample(texture[:,2,...].permute(0,3,1,2), xz).permute(0, 2, 3, 1)
        yz_feature = torch.nn.functional.grid_sample(texture[:,0,...].permute(0,3,1,2), yz).permute(0, 2, 3, 1)
        # #print("xy_feature shape : ",xy_feature.shape)
        self.xy_feature = None
        self.xz_feature = None
        self.yz_feature = None
        return xy_feature + xz_feature + yz_feature, toLight


 

    def fetch_normal_triplane(self, texture, dir):
        dir = cartesian_to_spherical_norm_torch(dir) * 2 - 1

        xy = torch.cat([dir[..., :1], dir[..., 1:2]], dim=-1)
        xz = torch.cat([dir[..., :1], dir[..., 2:3]], dim=-1)
        yz = torch.cat([dir[..., 1:2], dir[..., 2:3]], dim=-1)
    
        xz_feature = torch.nn.functional.grid_sample(texture[:,1,...].permute(0,3,1,2), xz,padding_mode="border").permute(0, 2, 3, 1)
        yz_feature = torch.nn.functional.grid_sample(texture[:,2,...].permute(0,3,1,2), yz,padding_mode="border").permute(0, 2, 3, 1)
        xy_feature = torch.nn.functional.grid_sample(texture[:,0,...].permute(0,3,1,2), xy,padding_mode="border").permute(0, 2, 3, 1)
        grid_visualize = ((dir * 0.5 + 0.5) * self.plane_cnt).int()
        grid_visualize = torch.where(grid_visualize % 2 == 0, grid_visualize.float()/ self.plane_cnt,1-  grid_visualize.float()/ self.plane_cnt)
        return xy_feature + xz_feature + yz_feature, grid_visualize
    def triplane_oct_sample(self, tri, xyz):
        r_norm, theta_norm, phi_norm = cartesian_to_spherical_norm_torch(xyz) 
        dir_norm = xyz / (xyz.norm(dim=-1, keepdim=True) + 1e-6)
        uv_octa = octahedral_project(dir_norm)

        uv_r_theta = torch.stack([r_norm, theta_norm], dim=-1)
        uv_r_phi   = torch.stack([r_norm, phi_norm],  dim=-1)
        uv_theta_phi = uv_octa

        B, _, W, H, C = tri.shape
        planes = tri.permute(0,1,4,2,3)
        p0, p1, p2 = planes[:,0], planes[:,1], planes[:,2]

        f0 = wrap_grid_sample(p0, uv_r_theta)
        f1 = wrap_grid_sample(p1, uv_r_phi)
        f2 = wrap_grid_sample(p2, uv_theta_phi)

        return f0 + f1 + f2
    # def fetch_normal_triplane_byvolume(self, texture, dir):
    #     dir = cartesian_to_spherical_norm_torch(dir) * 2 - 1
    #     xy = dir[:,0,:,:, :2].reshape(1,32,32,-1)
    #     xy2 = dir[:,1,:,:,:2].reshape(1,32,32,-1)
    #     print((xy2- xy).mean())
    #     xz = dir[:,:,0,:,[0,1]].reshape(1,32,32,-1)
    #     xz2 = dir[:,:,1,:,[0,1]].reshape(1,32,32,-1)
    #     print((xz2- xz).mean())
    #     yz = dir[:,0,:,:,1:].reshape(1,32,32,-1)
    #     yz2 = dir[:,1,:,:,1:].reshape(1,32,32,-1)
    #     print((yz2- yz).mean())
    #     xz_feature = torch.nn.functional.grid_sample(texture[:,1,...].permute(0,3,1,2), xz,padding_mode="border").permute(0, 2, 3, 1)
    #     yz_feature = torch.nn.functional.grid_sample(texture[:,2,...].permute(0,3,1,2), yz,padding_mode="border").permute(0, 2, 3, 1)
    #     xy_feature = torch.nn.functional.grid_sample(texture[:,0,...].permute(0,3,1,2), xy,padding_mode="border").permute(0, 2, 3, 1)
    #     grid_visualize = ((dir * 0.5 + 0.5) * self.plane_cnt).int()
    #     grid_visualize = torch.where(grid_visualize % 2 == 0, grid_visualize.float()/ self.plane_cnt,1-  grid_visualize.float()/ self.plane_cnt)
    #     return xy_feature,xz_feature,yz_feature,grid_visualize
    
    def fetch_traditional_triplane(self, texture, dir):
        dir = (dir - 0.1) / (7.67 - 0.1)
        dir = torch.clamp(dir,0,1) * 2 - 1
        dir = dir.repeat(1,1,1,1)
        xy = torch.cat([dir[..., :1], dir[..., 1:2]], dim=-1)
        xz = torch.cat([dir[..., :1], dir[..., 2:3]], dim=-1)
        yz = torch.cat([dir[..., 1:2], dir[..., 2:3]], dim=-1)
        xz_feature = torch.nn.functional.grid_sample(texture[:,1,...].permute(0,3,1,2), xz).permute(0, 2, 3, 1)
        yz_feature = torch.nn.functional.grid_sample(texture[:,2,...].permute(0,3,1,2), yz).permute(0, 2, 3, 1)
        xy_feature = torch.nn.functional.grid_sample(texture[:,0,...].permute(0,3,1,2), xy).permute(0, 2, 3, 1)
        grid_visualize = ((dir * 0.5 + 0.5) * self.plane_cnt).int()
        grid_visualize = torch.where(grid_visualize % 2 == 0, grid_visualize.float()/ self.plane_cnt,1-  grid_visualize.float()/ self.plane_cnt)
        return xy_feature + xz_feature + yz_feature, grid_visualize


class SimpleDiTransformerPlane(nn.Module):
    def __init__(self, configs,soft_max=False):
        super().__init__()
        self.embedding_size = configs["plane"]["encoder_direct_dim"]
        self.plane_config = configs["plane"]
        self.transport = configs["plane"]["transport"]
        if self.transport != "triplane_s":
            self.radius = 0.2
            self.ratio = self.plane_config["ratio"]
            self.plane_cnt = self.plane_config["resolution"]
            self.max_radius = 6
        else:
            self.radius = 0.2
            self.max_radius = 6
            self.plane_cnt = self.plane_config["resolution"]
            self.ratio = math.exp(math.log(self.max_radius / self.radius) / self.plane_cnt)
            self.max_base = math.pow(self.ratio,self.plane_cnt)
            # print(self.radius)
            # print(self.max_radius)
        self.up_scale = self.plane_config["up_scale"]
        self.min_distance = self.radius
        self.distance_range = self.max_radius - self.min_distance

        self.single_plane = False
        self.transformer_layer_cnt = configs["plane"]["transformer_cnt"]
        self.encoder_transformer_cnt = 6
        if "encoder_transformer_cnt" in configs["plane"].keys():
            self.encoder_transformer_cnt = configs["plane"]["encoder_transformer_cnt"]

        self.up_encoder = nn.ModuleList()

        embedding_size = configs["plane"]["encoder_direct_dim"]
        for i in range(self.up_scale):
            self.up_encoder.append(nn.Sequential(nn.Conv2d(embedding_size, embedding_size, 3, padding='same'), \
                                                 nn.Conv2d(embedding_size, embedding_size, 3, padding='same'), \
                                                 nn.LeakyReLU(),
                                                 nn.Conv2d(embedding_size, max(embedding_size // 2, 32), 1),
                                                 nn.LeakyReLU()))
            embedding_size = max(embedding_size // 2, 32)
        self.tri_plane_embed = nn.Parameter(
            0.01 * torch.randn(3, self.plane_cnt, self.plane_cnt, configs["plane"]["encoder_direct_dim"]),
            requires_grad=True)
        self.transformer_decoder = CrossAttentionBlock(inner_dim=self.embedding_size, cond_dim=self.embedding_size,
                                                       num_heads=16, eps=1e-6)

        self.output_layer = fc_layer(configs["plane"]["encoder_direct_dim"], configs["plane"]["decoder_direct_dim"])
        self.decoder_dim = configs["plane"]["decoder_direct_dim"]
        self.radiance_linear_layer_list = nn.ModuleList()

    def up_sample(self, photon_texture):
        for layer in self.up_encoder:
            photon_texture = F.interpolate(photon_texture, scale_factor=2)
            photon_texture = layer(photon_texture)
            photon_texture = photon_texture
        return photon_texture

    def sample_d(self, r, radius):
        max_radius = radius * self.max_base
        print(radius,max_radius)
        if self.plane_config["sample_type"] == "log":
            k = torch.log(r / radius) / math.log(self.ratio)
            k_int = torch.floor(k)
            is_r_less_than_r0 = r < radius

            is_r_lager_than_rmax = r > self.max_radius
            radius_small = radius * torch.pow(self.ratio, k_int)
            radius_large = radius * torch.pow(self.ratio, k_int + 1)
            a = 1 / (self.plane_cnt - 1)
            # #print(r.shape)
            # #print(self.max_radius)
            # normalize_r = ((r - radius_small) / (radius_large - radius_small)).reshape(512, 512, 1)
            # ####pyexr.write("../testData/normalized_r_{}.exr".format(self.idx),numpy.array(normalize_r.detach().cpu()))

            linear_r = (r - radius_small) / (radius_large - radius_small) * a + k_int * a
            linear_r[is_r_less_than_r0] = 0
            linear_r[is_r_lager_than_rmax] = 1
            # ####pyexr.write("../testData/linear_{}.exr".format(self.idx), numpy.array(r.reshape(512,512,1).detach().cpu()))

        else:
            linear_r = (r - radius) / (max_radius - radius)
        return linear_r.unsqueeze(dim=-1)

    def forward_triplane(self, photon_texture):
        B = photon_texture.shape[0]
        query = self.tri_plane_embed.reshape(-1, self.embedding_size).unsqueeze(0).repeat(B,1,1)

        photon_texture = photon_texture.reshape(B, -1, self.embedding_size)

        triplane_feature = self.transformer_decoder(query, photon_texture)
        # print("decoder")
        # print(triplane_feature.min())
        # print(triplane_feature.max())
        # #print(triplane_feature.shape)
        triplane_feature = triplane_feature.reshape(B,3, self.plane_cnt, self.plane_cnt, self.embedding_size)
        triplane_feature = self.output_layer(triplane_feature)

        return triplane_feature.permute(0,1, 4, 2, 3)

    def fetch_tri_plane(self, dir):
        toLight = dir / 8
        xy = torch.cat([toLight[..., :1], toLight[..., 1:2]], dim=-1)
        xz = torch.cat([toLight[..., :1], toLight[..., 2:3]], dim=-1)
        yz = torch.cat([toLight[..., 1:2], toLight[..., 2:3]], dim=-1)
        xy_feature = torch.nn.functional.grid_sample(self.xy_feature, xy).permute(0, 2, 3, 1)
        xz_feature = torch.nn.functional.grid_sample(self.xz_feature, xz).permute(0, 2, 3, 1)
        yz_feature = torch.nn.functional.grid_sample(self.yz_feature, yz).permute(0, 2, 3, 1)
        # #print("xy_feature shape : ",xy_feature.shape)
        self.xy_feature = None
        self.xz_feature = None
        self.yz_feature = None
        return xy_feature + xz_feature + yz_feature, None

    def fetch_spherical_tri_plane(self, triplane, dir,radius,channel_cnt):
        xy_feature = triplane[:,0, ...]
        xz_feature = triplane[:,1, ...]
        yz_feature = triplane[:,2, ...]
        shape_len = len(dir.shape)
        toLight = dir
        B, W, H, C = toLight.shape
        toLight = toLight.reshape(-1, W, H, C)
        toLightSquareDistance = torch.sum((toLight * toLight), dim=-1)
        toLightDistance = torch.sqrt(toLightSquareDistance)
        voxel_d = self.sample_d(toLightDistance,radius)
        # print(voxel_d.min())
        # print(voxel_d.max())
        # pyexr.write("../testData/voxel_d.exr",voxel_d[0].cpu().numpy())
        voxel_d = voxel_d * 2 - 1
        toLightNormalization = toLight / toLightDistance.unsqueeze(dim=-1)
        uv = EqualAreaSphereToSquare(toLightNormalization)  ## for
        u = uv[..., 1].clone()
        uv[..., 1] = uv[..., 0]
        uv[..., 0] = u
        voxel_coord = torch.cat([uv, voxel_d], dim=-1)
        xy = torch.cat([voxel_coord[..., :1], voxel_coord[..., 1:2], voxel_coord[..., 1:2]], dim=-1)
        xz = torch.cat([voxel_coord[..., :1], voxel_coord[..., 2:3]], dim=-1)
        yz = torch.cat([voxel_coord[..., 1:2], voxel_coord[..., 2:3]], dim=-1)
        xy[..., 2:3] = 0
        xy_feature = grid_sample_n.apply(xy_feature.unsqueeze(2),
                                         xy.unsqueeze(dim=1)).squeeze(dim=2).permute(0,2,3,1)
        xz_feature = torch.nn.functional.grid_sample(xz_feature, xz).permute(0, 2, 3, 1)
        yz_feature = torch.nn.functional.grid_sample(yz_feature, yz).permute(0, 2, 3, 1)

        # #print(xy_feature.shape)
        voxel_coord = torch.floor((voxel_coord + 1) /2 * self.plane_cnt)
        return xy_feature + xz_feature + yz_feature, voxel_coord

class PfGL_Encoder_Fetch(nn.Module):  ##grid sampling first then  generate triplane feature by transformer
    def __init__(self, configs):
        super().__init__()
        self.input_dim = configs["light_encoders"]["photon_vpls_encoder"]["input_dim"]
        if configs["light_encoders"]["photon_vpls_encoder"]["type"] == "autoencoder":
            self.net = EncoderFormView(in_channels=configs["light_encoders"]["photon_vpls_encoder"]["input_dim"],
                                       ch=configs["encoder_direct_dim"])
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "patchencoder":
            self.net = PatchEncoder(in_channels=configs["light_encoders"]["photon_vpls_encoder"]["input_dim"])
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "vaeencoder":
            self.net = VaeEncoder(in_channels=configs["vae_input_dim"], z_channels=configs["vae_output_dim"])
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "mlpencoder":
            self.net = MlpEncoder(configs["encoder_direct_dim"])
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "transformer":
            self.net = MlpTransformer(configs["encoder_direct_dim"])
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "multiscale":
            self.net = MultiScaleTransformer(configs["encoder_direct_dim"])
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "planeencoder":
            self.net = PlaneEncoder(configs["encoder_direct_dim"])
        else:
            self.net = None
        # self.aux_net = AuxMlpTransformer(configs["encoder_direct_dim"])
        if configs["conenet"] != "none":
            self.cone_net = eval(
                "{}(configs,configs[\"specular_dim\"],configs[\"specular_level\"])".format(configs["conenet"]))
        else:
            self.cone_net = None
        self.space_weight_layer = nn.Sequential(fc_layer(3, 16),
                                                fc_layer(16, 16),
                                                fc_layer(16, 16),
                                                fc_layer(16, configs["specular_level"]))
        self.view_weight_layer = nn.Sequential(fc_layer(1, 16),
                                               fc_layer(16, 16),
                                               fc_layer(16, 16),
                                               fc_layer(16, configs["specular_level"]))
        self.specular_level = configs["specular_level"]
        self.specular_dim = configs["specular_dim"]
        self.sampleCnt = 1000
        self.init_scale = 1e-2
        self.plane = DiTransformerPlane(configs)
        self.grid_dim = configs["plane"]["grid_dim"]
        self.transport = configs["plane"]["transport"]
        self.z_visualize = configs["z_visualize"]
        self.step_cnt = 0
        self.tanh = math.tan(configs["fov"] / 180 * math.pi)
        self.train_set = configs["train_set"]
        # self.specular_encoder = nn.Sequential(
        #     fc_layer(self.specular_dim + 2,self.specular_dim),
        #     fc_layer(self.specular_dim ,self.specular_dim),
        #     fc_layer(self.specular_dim ,self.specular_dim)
        # )

    def get_first_input(self, data):
        B, W1, H1, W2, H2, C = data["global"]["position"].shape
        # input = torch.cat([data["global"]["position"],data["global"]["direction"],data["global"]["radiance"]],dim=-1).reshape(-1,W2,H2,9)
        input = data["global"]["radiance"].reshape(-1, W2, H2, 3).unsqueeze(-1)
        input = input.permute(3, 0, 1, 2, 4).reshape(B * W1 * H1 * 3, W2, H2, 1)
        return input

    def get_second_input(self, data):
        B, W1, H1, W2, H2, C = data["global"]["position"].shape
        # input = torch.cat([data["global"]["position"],data["global"]["direction"],data["global"]["radiance"]],dim=-1).reshape(-1,W2,H2,9)
        input1 = data["global"]["radiance"].reshape(-1, W2, H2, 3)
        input2 = data["global"]["position"].reshape(-1, W2, H2, 3)
        input3 = data["global"]["direction"].reshape(-1, W2, H2, 3)
        input = torch.cat([input1, input2, input3], dim=-1)
        return input

    def get_4_input(self, data):
        B, W1, H1, W2, H2, C = data["global"]["position"].shape
        # input = torch.cat([data["global"]["position"],data["global"]["direction"],data["global"]["radiance"]],dim=-1).reshape(-1,W2,H2,9)
        input1 = data["global"]["radiance"].reshape(-1, W2, H2, 3).unsqueeze(-1)
        input2 = data["global"]["depth"].reshape(-1, W2, H2, 1).unsqueeze(-1).repeat(1, 1, 1, 3, 1)
        input = torch.cat([input1, input2], dim=-1)
        input = input.permute(3, 0, 1, 2, 4).reshape(B * W1 * H1 * 3, W2, H2, 2)
        return input

    def get_7_input(self, data):
        B, W1, H1, W2, H2, C = data["global"]["position"].shape
        # input = torch.cat([data["global"]["position"],data["global"]["direction"],data["global"]["radiance"]],dim=-1).reshape(-1,W2,H2,9)
        input1 = data["global"]["radiance"].reshape(-1, W2, H2, 3).unsqueeze(-1)
        input2 = data["global"]["depth"].reshape(-1, W2, H2, 1).unsqueeze(-1).repeat(1, 1, 1, 3, 1)
        # #print(data["global"].keys())
        dir = data["global"]["direction"].reshape(-1, W2, H2, 3).unsqueeze(-2).repeat(1, 1, 1, 3, 1)

        # #print("direct get dir encoder dim ",dir.shape)
        input = torch.cat([input1, input2, dir], dim=-1)
        input = input.permute(3, 0, 1, 2, 4).reshape(B * W1 * H1 * 3, W2, H2, 5)  # 3 , 32, 32 ,64 ,64 ,5
        return input

    def fetch(self, dir):
        if self.transport == "triplane":
            return self.plane.fetch_tri_plane(dir)
        elif self.transport == "triplane_s":
            return self.plane.fetch_spherical_tri_plane(dir)
        else:
            return self.plane.fetch_grid(dir)

    def visualize(self, photon_texture):
        min = photon_texture.min()
        max = photon_texture.max()
        visualize_texture = (photon_texture - min) / (max - min)
        B, W1, H1, C = photon_texture.shape
        visualize_map = torch.zeros(64, 64, 3)
        for i in range(B):
            for w1 in range(W1):
                for h1 in range(H1):
                    startx = w1 * 4
                    starty = h1 * 4
                    visualize_map[startx:startx + 4, starty:starty + 4, ...] = photon_texture[i, w1, h1].reshape(4, 4,
                                                                                                                 3)
            ####pyexr.write("../testData/photon_texture{}.exr".format(i), numpy.array(visualize_map.detach().cpu()))

    def specular(self, data, space_texture, direction_texture, radiance_texture):
        specular_reflect_dir = torch.sum(data["local"]["gbuffer"]["view_dir"] * data["local"]["gbuffer"]["normal"],
                                         dim=-1).unsqueeze(-1) * data["local"]["gbuffer"]["normal"] * 2 - \
                               data["local"]["gbuffer"]["view_dir"]
        specular_reflect_dir_dot = specular_reflect_dir[..., :1]
        space_weight = self.space_weight_layer(torch.cat(
            [data["local"]["gbuffer"]["roughness"], data["local"]["toLight"][..., 3:4], specular_reflect_dir_dot],
            dim=-1))
        view_weight = self.view_weight_layer(torch.cat([data["local"]["gbuffer"]["roughness"]], dim=-1))
        space_weight = torch.softmax(space_weight, dim=-1)
        view_weight = torch.softmax(view_weight, dim=-1)
        data["local"]["light_depth"] = data["local"]["toLight"][..., 3:4]
        data["local"]["space_weight"] = space_weight[..., :min(self.specular_level, 4)]
        data["local"]["view_weight"] = view_weight[..., :min(self.specular_level, 4)]
        bias = data["local"]["bias"]
        bias = bias / self.tanh  # 归一化到uv空间f
        # #print(self.tanh)
        limit = 1
        result = torch.zeros(3, 512, 512, self.specular_dim).cuda()
        # #print(data["local"]["reflect_mask"].shape)
        # #print(bias.shape)
        # exit()
        mask = data["local"]["reflect_mask"].bool() | (bias[..., 0:1] > limit).repeat(1, 1, 1, 3) | (
                bias[..., 1:2] > limit).repeat(1, 1, 1, 3) | (bias[..., 0:1] < -limit).repeat(1, 1, 1,
                                                                                              3) | (
                       bias[..., 1:2] < -limit).repeat(1, 1, 1, 3)
        cha = (torch.abs(bias) - limit)
        data["local"]["cha"] = torch.where(cha < 0, 0, cha)
        # compute reflect dir

        bias = torch.where(bias > limit, limit, bias)
        bias = torch.where(bias < -limit, -limit, bias)
        bias = bias.expand(3, 512, 512, 2)
        i = 0
        # #print(bias.shape)
        # #print("drection_texture",direction_texture.shape)
        direction_feature = torch.zeros((3, 512, 512, self.specular_dim // 2)).cuda()
        space_feature = torch.zeros((3, 512, 512, self.specular_dim // 2)).cuda()
        for i in range(self.specular_level):
            direction_feature = direction_feature + torch.nn.functional.grid_sample(direction_texture[i], bias,
                                                                                    padding_mode="reflection").permute(
                0, 2, 3, 1) * view_weight[..., i:i + 1]

            # for texture in reflect_texture:
        for i in range(self.specular_level):
            B, C, W, H = space_texture[i].shape
            uv_ll, uv_rl, uv_lr, uv_rr, uv_weight = oct_transform(data["local"]["space_uv"], W)
            uv_weight = uv_weight.permute(0, 3, 1, 2)
            # #print("space_texture {} ".format(i),space_texture[i].shape)
            space_feature_lu = torch.nn.functional.grid_sample(space_texture[i],
                                                               uv_ll.expand(3, 512, 512, 2)) * uv_weight[:, :1, ...]
            space_feature_ru = torch.nn.functional.grid_sample(space_texture[i],
                                                               uv_rl.expand(3, 512, 512, 2)) * uv_weight[:, 1:2, ...]
            space_feature_lt = torch.nn.functional.grid_sample(space_texture[i],
                                                               uv_lr.expand(3, 512, 512, 2)) * uv_weight[:, 2:3, ...]
            space_feature_rt = torch.nn.functional.grid_sample(space_texture[i],
                                                               uv_rr.expand(3, 512, 512, 2)) * uv_weight[:, 3:4, ...]
            space_feature = space_feature + (
                    space_feature_lu + space_feature_ru + space_feature_lt + space_feature_rt).permute(0, 2, 3,
                                                                                                       1) * space_weight[
                                                                                                            ...,
                                                                                                            i:i + 1]

        data["local"]["reflect_feature"] = torch.cat([space_feature, direction_feature], dim=-1)
        #     feature = feature_lu * uv_weight[:,:1,...] + feature_ru * uv_weight[:,1:2,...]+ feature_lt * uv_weight[:,2:3,...]+ feature_rt * uv_weight[:,3:4,...]
        #     if i != 4:
        #         # #print("weight:",weight[...,i:i+1].repeat(3,1,1,1).shape)
        #         # #print("feature:",feature.permute(0,2,3,1).shape)
        #         #exit()
        #         result =result + weight[...,i:i+1].repeat(3,1,1,1) * feature.permute(0,2,3,1)
        #     else:
        #         data["local"]["direct_light_reprs"] = self.specular_encoder(torch.cat([feature.permute(0,2,3,1),cha.repeat(3,1,1,1)],dim=-1))
        #     i = i + 1
        # data["local"]["reflect_feature"] = result
        # i = 0
        # for texture in radiance_texture:
        #     feature_lu = torch.nn.functional.grid_sample(texture.permute(0, 3, 1, 2), uv_ll)
        #     feature_ru = torch.nn.functional.grid_sample(texture.permute(0, 3, 1, 2), uv_rl)
        #     feature_lt = torch.nn.functional.grid_sample(texture.permute(0, 3, 1, 2), uv_lr)
        #     feature_rt = torch.nn.functional.grid_sample(texture.permute(0, 3, 1, 2), uv_rr)
        #     feature = feature_lu * uv_weight[:,:1,...] + feature_ru * uv_weight[:,1:2,...]+ feature_lt * uv_weight[:,2:3,...]+ feature_rt * uv_weight[:,3:4,...]
        #     data["local"]["reflect_texture_{}".format(i)] = feature.permute(0,2,3,1)
        #     data["local"]["radiance_texture_{}".format(i)] = texture
        #     # #print("radiance texture :",uv_ll.device)
        #     # exit()
        #     #data["local"]["radiance_texture_{}".format(i)][mask] =0
        #     i= i + 1

        return None

    def transform_fov(self, input):
        full = (int(64 * self.tanh) + 1) // 2 * 2
        cha = (full - 64) // 2
        padding_size = (cha, cha, cha, cha)
        input = input.permute(0, 3, 1, 2)
        input = torch.nn.functional.pad(input, padding_size, "constant", 0)
        input = nn.functional.interpolate(input, scale_factor=64 / full)
        return input.permute(0, 2, 3, 1)

    def forward(self, data, enable_timing_profile=False):

        inputList = []
        timing_profile = {}
        if self.input_dim == 3:
            input1 = self.get_first_input(data)
        elif self.input_dim == 9:
            input1 = self.get_second_input(data)
        elif self.input_dim == 4:
            input1 = self.get_4_input(data)
        elif self.input_dim == 7:
            input1 = self.get_7_input(data)
        aux_input = self.get_7_input(data)
        # #print("direct get input encoder dim ",input1.shape)
        # #print(input1.shape)
        B, W1, H1, C = input1.shape
        width = int(math.sqrt(B / 3))
        # with torch.no_grad():
        if self.net != None:
            photon_texture = self.net(input1).reshape(3, width, width, -1)
            # aux_texture = self.aux_net(aux_input).reshape(3,width,width,-1)
        index = int(data["local"]["index"][0])
        # space_texture,direction_texture,radiance_texture = self.cone_net(input1,index)
        if self.cone_net != None:
            input1 = self.transform_fov(input1)
            data["global"]["radiance"] = input1[..., :1].reshape(3, 32, 32, 64, 64).permute(1, 2, 3, 4, 0).unsqueeze(0)
            space_texture, direction_texture, radiance_texture, pred_radiance, clip_radiance = self.cone_net.forward(
                input1, index)
            if self.train_set["need_specular"]:
                self.specular(data, space_texture, direction_texture, radiance_texture)
            data["local"]["pred_clip_radiance"] = pred_radiance
            data["local"]["clip_radiance"] = clip_radiance
        # #print(photon_texture.shape)
        # data["photon_texture"] = photon_texture
        # self.visualize(photon_texture)
        if self.transport == "residual":
            self.plane.forward_grid(data, photon_texture)
        elif self.transport == "transformer" or self.transport == "standard":
            self.plane.forward_no_residual(data, photon_texture, None)
        elif self.transport == "triplane" or self.transport == "triplane_s":
            self.plane.forward_triplane(data, photon_texture, None)
        elif self.transport == "unet":
            self.plane.forward_depth(data, photon_texture)
        elif self.transport == "grid":
            self.plane.voxel_grid = self.plane.direct_grid[:, data["local"]["index"][0], ...].squeeze(1)
            # print("voxel_grid_shape: ", self.plane.voxel_grid.shape)
            self.step_cnt = self.step_cnt + 1
            if self.step_cnt % 4000 == 0:
                grid = self.plane.voxel_grid.detach().cpu().numpy()
                with gzip.open('./grid.pkl.gz', 'wb') as f:
                    pickle.dump(grid, f, pickle.HIGHEST_PROTOCOL)
                f.close()
        else:
            None
        return


class TPfGL_Encoder(nn.Module):  ## generate triplane feature by transformer first then grid sampling
    def __init__(self, configs):
        super().__init__()
        if configs["light_encoders"]["photon_vpls_encoder"]["type"] == "mlp":
            self.net = modules.MLP(
                configs["light_encoders"]["photon_vpls_encoder"]['dims'],
                'lrelu',
                'none')
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "pointnet":
            self.net = PointNet(configs["light_encoders"]["photon_vpls_encoder"])
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "pointnet2":
            self.net = PointNet2(configs["light_encoders"]["photon_vpls_encoder"])
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "pointnet3":
            self.net = PointNet3(configs["light_encoders"]["photon_vpls_encoder"])
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "pointnet4":
            self.net = PointNet4(configs["light_encoders"]["photon_vpls_encoder"])
        elif configs["light_encoders"]["photon_vpls_encoder"]["type"] == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(configs["light_encoders"]["photon_vpls_encoder"]['dims'],
                                                       nhead=configs["light_encoders"]["photon_vpls_encoder"]['head'])
            self.net = nn.TransformerEncoder(encoder_layer, 6)
        self.photon_repr = configs["light_encoders"]["photon_vpls_encoder"]["repr_dim"]
        self.sampleCnt = 1000
        self.shading_clue_encoders = Shading_Encoders(configs['shading_encoders'])
        self.gbuffer_encoder = modules.MLP(
            [10, configs["light_encoders"]["photon_vpls_encoder"]["encoder_feat_dim"],
             configs["light_encoders"]["photon_vpls_encoder"]["encoder_feat_dim"]],
            'lrelu',
            'none'
        )
        self.init_scale = 1e-2
        self.posTriplane = VanillaTriplane()

    def forward(self, data, enable_timing_profile=False):

        local_data = data['local']
        inputList = []
        timing_profile = {}
        if enable_timing_profile:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
        B, L, W, H, _ = local_data["toLight"].shape
        W2, H2 = (32, 32)
        photon_texture = self.net(data)

        if enable_timing_profile:
            ender.record()
            torch.cuda.synchronize()
            timing_profile['direct_light_encoding'] = starter.elapsed_time(ender)
            # print("Photon Encoder Time ： {}s", timing_profile['direct_light_encoding'] / 1000.0)
        shading_clue_reprs, shading_clue_timing_profile = self.shading_clue_encoders(data['lights'], local_data,
                                                                                     enable_timing_profile)
        pos_embed = self.posTriplane.plane_pos_embed
        pos_embed = pos_embed.reshape(B * L, W * H, -1)
        ## TRICK
        cond_embed = photon_texture.reshape(B * L, W2 * H2, -1)

        light_field_feature = self.transformer(pos_embed, cond=cond_embed).reshape(B, L, W, H, -1)
        return data, timing_profile, light_field_feature, shading_clue_reprs


class TPfGL_Full(NeGL_Base):
    def __init__(self, configs, loss_configs=None):
        super().__init__(configs, loss_configs, load_decoder=True)
        # Light encoders
        self.negl_direct = PfGL_Encoder(configs)
        self.configs = configs
        self.gbuffer_mlp = modules.MLP(configs["gbuffer_mlp"]["dims"], "relu", "relu")
        self.cross_attention_lights = modules.CrossAttention(self.configs["pointnet"]["embed_dim"], 10,
                                                             self.configs["pointnet"]["embed_dim"])

        self.shading_clue_encoders = Shading_Encoders(configs['shading_encoders'])
        self.final_linear = nn.Linear(self.negl_direct.photon_repr, self.negl_direct.photon_repr, bias=True).cuda()
        self.negl_indirect = NeGL_Indirect(configs)
        ##self.negl_indirect = NeGL_Indirect(configs)

    def cut_patch(self, data, enable_timing):
        u = random.uniform(0, 1)
        v = random.uniform(0, 1)
        x = (int)(512 - 16) * u
        y = (int)(512 - 16) * v

    def forward(self, data, enable_timing_profile=False, train_setting=None):
        need_direct, need_shadow, need_indirect = (True, True, True)
        if train_setting != None:
            need_direct, need_shadow, need_indirect = (
                train_setting["need_direct"], train_setting["need_shadow"], train_setting["need_indirect"])
        global_timing_profile = {}

        localData = data["local"]
        ##data, timing_indirect = self.negl_indirect(data, enable_timing_profile)
        if enable_timing_profile:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
        if need_direct:
            data_direct, timing_profile, direct_light_reprs = self.negl_direct(data, enable_timing_profile)
            gbuffer = utils.get_GB_features(localData, self.configs["shading_decoders"]["direct_decoder"]["input"])
            light_position = data["lights"]["direct_vpls"]["position"].mean(dim=-2)
            B, L, W, H, C = direct_light_reprs.shape
            _, W, H, C2 = gbuffer.shape
            if self.configs["pointnet"]["type"] == "triplane_cross":
                attention_input = direct_light_reprs.permute(0, 2, 3, 1, 4)
                buffer = utils.get_GB_features(localData, self.configs["pointnet"]["input"])
                attention_input = attention_input.reshape(-1, L, C)
                buffer = buffer.reshape(-1, 1, buffer.shape[-1])
                attention_lights, attention_weights = self.cross_attention_lights(attention_input, buffer)

                data["light_repr_photon_vpls"] = self.final_linear(attention_lights).reshape(B, W, H, -1)
                attention_weights = attention_weights.reshape(B, W, H, L, -1).permute(0, 3, 1, 2, 4)
                data["attention_weight"] = attention_weights
            else:
                raise NotImplementedError(self.configs["pointnet"])
        shading_clue_reprs, shading_clue_timing_profile = self.shading_clue_encoders(localData,
                                                                                     enable_timing_profile)
        if enable_timing_profile:
            starter.record()
        result = self.shading_decoder.forward(data, train_setting)
        if enable_timing_profile:
            ender.record()
            torch.cuda.synchronize()
            timing_profile['deocder'] = starter.elapsed_time(ender)
            # print("decoder time {}", timing_profile['deocder'] / 1000.0)

        # Calculate loss
        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)

        return result, loss_map, global_timing_profile


class RandomViewFull(NeGL_Base):
    def __init__(self, configs, loss_configs=None):
        super().__init__(configs, loss_configs, load_decoder=True)
        # Light encoders
        self.negl_direct = PfGL_Encoder(configs)
        self.auto_decoder = Decoder()

    def forward(self, data, enable_timing_profile=False, train_setting=None):
        data_direct, timing_profile, direct_light_reprs = self.negl_direct(data, enable_timing_profile)
        B, N, C = direct_light_reprs.shape
        data["direct_light_reprs"] = direct_light_reprs
        direct_light_reprs = direct_light_reprs.reshape(B * N, 4, 4, 3)
        value = self.auto_decoder(direct_light_reprs)
        value = value.reshape(B * N, 16, 16, 3)
        result = {
            "log1p_radiance": value,
            "radiance": expp1_torch(value)
        }
        data["local"] = {}
        data["local"]["radiance"] = data["radiance"].reshape(value.shape)
        data["local"]["log1p_radiance"] = data["log1p_radiance"].reshape(value.shape)
        # Calculate loss
        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)

        return data, result, loss_map, {}


class RandomViewFullComponent(nn.Module):
    def __init__(self, configs, loss_configs=None):
        super().__init__()
        # Light encoders
        self.configs = configs
        self.negl_direct = PfGL_Encoder_Fetch(configs)
        self.auto_decoder = Decoder()

    def forward(self, data, enable_timing_profile=False, train_setting=None):
        self.negl_direct(data, enable_timing_profile)
        # #print("local toLight datatype:",data["local"]["toLight"].dtype)
        if True:
            gbuffer_direct_light_reprs, coord = self.negl_direct.fetch(
                -data["local"]["toLight"][..., :3] * data["local"]["toLight"][..., 3:4])
            data["local"]["direct_light_reprs"] = gbuffer_direct_light_reprs
        data["local"]["shadow_light_reprs"] = None
        data["local"]["voxel_d"] = coord
        return data, {}, {}


def get_forward_up_right_tensor(to_light):
    to_light = normalize(to_light)
    up = torch.zeros_like(to_light)
    up[..., :] = 0
    up[..., 2:3] = 1
    right = normalize(torch.cross(to_light, up, dim=-1))
    up = normalize(torch.cross(to_light, right, dim=-1))
    return to_light, up, right

def get_forward_up_right_tensor_for_indirect(to_light):
    up = torch.zeros_like(to_light)
    up[..., :] = 0
    up[..., 2:3] = 1
    right = normalize(torch.cross(to_light, up, dim=-1))
    up = normalize(torch.cross(to_light, right, dim=-1))
    return to_light, -up, -right


def rotate(a, b, c, v):
    return torch.cat([torch.sum((a * v), dim=-1).unsqueeze(-1),
                      torch.sum((b * v), dim=-1).unsqueeze(-1),
                      torch.sum((c * v), dim=-1).unsqueeze(-1)], dim=-1)


class RsmIndirectNetwork(nn.Module):
    def __init__(self, configs, loss_configs=None, train_setting=None):
        super().__init__()  # 64  64 2 -> 16 16 2  8 8 4

        self.upsample = nn.Upsample(scale_factor=128 / 1280, mode='bilinear', align_corners=False)
        self.loss_func = Loss(loss_configs) if loss_configs else None
        self.indirect_gbuffer_feature = ["N", "LP", "R", "V"]
        self.rsm_gbuffer_feature = ["N", "LP", "R", "V", "D", "S"]
        self.indirect_repr = configs["indirect_repr"]
        self.is_channel_cut = configs["is_channel_cut"]
        self.screen_view_repr_dim = configs["scene_repr"]

        self.light_input_dim = 6
        self.light_output_dim = 3
        self.light_view_input_dim = 9
        self.gbuffer_dim = 10
        if self.is_channel_cut:
            self.light_input_dim = 4
            self.light_output_dim = 1
            self.light_view_input_dim = 7
            self.gbuffer_dim = 10
        self.light_encoder = nn.Sequential(
            fc_layer(self.light_view_input_dim, self.indirect_repr),
            fc_layer(self.indirect_repr, self.indirect_repr),
            fc_layer(self.indirect_repr, self.indirect_repr),
            fc_layer(self.indirect_repr, self.indirect_repr)
        )
        self.scene_encoder = NewSUNet(10, self.screen_view_repr_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.indirect_repr, num_heads=8, kdim=self.indirect_repr, vdim=self.indirect_repr,
            batch_first=True)
        self.light_input_layer = nn.Linear(self.indirect_repr, self.indirect_repr)
        self.decoder = nn.Sequential(
            fc_layer(self.indirect_repr + self.gbuffer_dim + 2, self.indirect_repr),
            fc_layer(self.indirect_repr, self.indirect_repr),
            fc_layer(self.indirect_repr + self.gbuffer_dim + 2, self.indirect_repr),
            fc_layer(self.indirect_repr, 1)
        )

    def get_lightformer_input(self, localLightData, localData):
        # #print(localLightData['shadow'].keys())
        z_f = localLightData['shadow']['pixel_emitter_distance'] / 3
        z = localLightData['shadow']['occluder_emitter_distance'] / 3

        position_mask = localData["position_mask"]
        z_f[position_mask[..., 0]] = 3
        z[position_mask[..., 0]] = -3

        normal = localData["gbuffer"]["normal"]

        toLight = localData["toLight"][..., :3]
        toView = localData["gbuffer"]["view_dir"]
        c_c = torch.sum(localData["gbuffer"]['normal'] * localData["gbuffer"]['view_dir'], axis=-1)[
            ..., None]  # diffuse # TODO: Use this feature?
        depth = localData["gbuffer"]["depth"]
        return torch.cat([z_f - z, z_f / z, depth, c_c], dim=-1)

    def get_camera_buffer(self, localData):
        camera_pos = localData["camera_pos"]
        forward = -localData["gbuffer"]["view_dir"][0, 255, 255]
        reflect_forward, reflect_up, reflect_right = get_forward_up_right_tensor_for_indirect(forward)
        camera_to_gbuffer = localData["gbuffer"]["position"] - camera_pos

        localData["gbuffer"]["cposition"] = rotate(-reflect_right, reflect_up, reflect_forward, camera_to_gbuffer)
        localData["gbuffer"]["cnormal"] = rotate(-reflect_right, reflect_up, reflect_forward,
                                                 localData["gbuffer"]["normal"])
        # ####pyexr.write("/seaweedfs_tmp/training/wangjiu/new/general_shading/testData/camera_space/camera_position.exr",localData["gbuffer"]["cposition"][0].detach().cpu().numpy())
        # #print("succeed")
        # exit()
        return localData

    def forward(self, data, enable_timing_profile=False, train_setting=None):
        data["local"] = self.get_camera_buffer(data["local"])
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        # indirect_feature = self.light_encoder(indirect_repr_input).reshape(-1,W1 * W1 * W2  *W2 ,self.indirect_repr).mean(dim=1)
        if self.is_channel_cut:
            indirect_gbuffers = utils.get_GB_features_cut_channel(data["local"]["gbuffer"],
                                                                  self.indirect_gbuffer_feature)
            rsm_gbuffers = utils.get_GB_features_cut_channel(data["local"]["gbuffer"], self.rsm_gbuffer_feature)
            light_position = self.upsample(
                data["local"]["lights"]["shadow"]["light_position"].permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(
                -2).repeat(1, 1, 1, 3, 1).permute(3, 0, 1, 2, 4).squeeze(1)
            light_albedo = self.upsample(data["local"]["lights"]["shadow"]["light_albedo"].permute(0, 3, 1, 2)).permute(
                0, 2, 3, 1).unsqueeze(-1).permute(3, 0, 1, 2, 4).squeeze(1)
            light_normal = self.upsample(data["local"]["lights"]["shadow"]["light_normal"].permute(0, 3, 1, 2)).permute(
                0, 2, 3, 1).unsqueeze(-2).repeat(1, 1, 1, 3, 1).permute(3, 0, 1, 2, 4).squeeze(1)
        else:
            indirect_gbuffers = utils.get_GB_features_channel(data["local"]["gbuffer"], self.indirect_gbuffer_feature)
            rsm_gbuffers = utils.get_GB_features_channel(data["local"]["gbuffer"], self.rsm_gbuffer_feature)
            light_position = self.upsample(
                data["local"]["lights"]["shadow"]["light_position"].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            light_albedo = self.upsample(data["local"]["lights"]["shadow"]["light_albedo"].permute(0, 3, 1, 2)).permute(
                0, 2, 3, 1)
            light_normal = self.upsample(data["local"]["lights"]["shadow"]["light_normal"].permute(0, 3, 1, 2)).permute(
                0, 2, 3, 1)

        light_input = torch.cat([light_position, light_albedo, light_normal], dim=-1)
        # print("light position shape", light_position.shape)
        light_repr = self.light_encoder(light_input).reshape(-1, 128 * 128, self.indirect_repr).mean(dim=1)
        geo_repr = self.scene_encoder(indirect_gbuffers.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        value = light_repr.unsqueeze(1)
        key = self.light_input_layer(value)
        # print("value shape", value.shape)
        # print("geo shape", geo_repr.shape)
        light_feature = self.cross_attention(geo_repr.reshape(3, -1, self.indirect_repr), key, value)[0]
        light_feature = light_feature.reshape(3, 512, 512, -1)
        feature = light_feature
        i = 0
        for layer in self.decoder:
            if i % 2 == 0:
                feature = torch.cat([feature, rsm_gbuffers], dim=-1)
            feature = layer(feature)
            i = i + 1
        indirect_shading = feature
        # print("indirect_shading shape", indirect_shading.shape)
        if self.is_channel_cut:
            indirect_shading = indirect_shading.permute(3, 1, 2, 0)
        localData = data["local"]
        result = {}
        result["log1p_indirect_shading"] = torch.where(localData["mask"], localData["log1p_indirect_shading"],
                                                       indirect_shading)
        result["indirect_shading"] = expp1_torch(result["log1p_indirect_shading"])

        result["diffuse_shading"] = data["local"]["diffuse_shading"]
        result["specular_shading"] = data["local"]["specular_shading"]
        result["shadow"] = data["local"]["shadow"]
        ender.record()
        torch.cuda.synchronize()
        # print("indirect inference time: ", starter.elapsed_time(ender))
        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)

        return data, result, loss_map, {}


class LightformerDecoder(nn.Module):
    def __init__(self, configs, loss_configs=None, train_setting=None):
        super().__init__()
        self.train_setting = configs["train_set"]
        self.light_encoder = nn.Sequential(
            fc_layer(9, 256),
            fc_layer(256, 256),
            fc_layer(256, 256),
            fc_layer(256, 256),
            nn.Linear(256, 256)
        )
        self.gbuffer_feature = ["N", "P", "R", "D", "S"]
        self.cross_attention_lights = modules.CrossAttention(256, 13,
                                                             256)
        self.shading_clue_encoders = Direct_Encoders_Decoder_Lightformer(configs)
        self.loss_func = Loss(loss_configs) if loss_configs else None
        self.light_dir_encoder = nn.Sequential(
            fc_layer(3,64),
            fc_layer(64,64),
            fc_layer(64,64),
            fc_layer(64,32),
        )
        self.light_dir_linear = nn.Linear(32,32)
        self.halfvec_encoder = nn.Sequential(
            fc_layer(3, 64),
            fc_layer(64, 64),
            fc_layer(64, 64),
            fc_layer(64, 32),
        )
        self.half_vec_linear = nn.Linear(32, 32)

    def down_sampleing_plane(self, light_input, angular_size, space_size):
        B, W1, H1, W2, H2, C = light_input.shape
        angular_pool = nn.AvgPool2d(angular_size)
        space_pool = nn.AvgPool2d(space_size)
        light_input = space_pool(light_input.permute(0, 1, 2, 5, 3, 4).reshape(B * W1 * H1, C, W2, H2))
        W2, H2 = W2 // space_size, H2 // space_size
        light_input = light_input.reshape(B, W1, H1, C, W2, H2).permute(0, 4, 5, 3, 1, 2).reshape(B * W2 * H2, C, W1,
                                                                                                  H1)
        light_input = angular_pool(light_input)
        W1, H1 = W1 // angular_size, H1 // angular_size
        light_input = light_input.reshape(B, W2, H2, C, W1, H1).permute(0, 4, 5, 1, 2, 3)
        return light_input
        
    def forward(self, data, enable_timing_profile=False, train_setting=None):
        light_radiance = data["global"]["radiance"]
        direction = data["global"]["direction"]
        position = data["global"]["position"]
        input = torch.cat([light_radiance, direction,position], dim=-1)
        input = self.down_sampleing_plane(input,2,1)
        print(input.shape)
        #exit()
        input = input.reshape(1,-1,9)
        light_repr = self.light_encoder(input).mean(dim=1).unsqueeze(1)
        data["local"]["lightdir_feature"] = self.light_dir_encoder(data["local"]["lights"]["specular"]["light_dir"])
        data["local"]["halfvec_feature"] = self.light_dir_encoder(data["local"]["lights"]["specular"]["half_vec"])
        # print("light_repr shape:", light_repr.shape)
        # exit()
        gbuffer = utils.get_GB_features(data["local"]["gbuffer"], self.gbuffer_feature)
        # print_image_exr(gbuffer[...,-3:],"S")
        # print_image_exr(gbuffer[...,-6:-3],"D")
        light_repr,weights = self.cross_attention_lights(light_repr,gbuffer.reshape(1,-1,13))
        _,W,H,_ = data["local"]["direct_shading"].shape
        light_repr = light_repr.reshape(1,W,H,256)
        weights = weights.reshape(1,W,H,1)
        data["local"]["light_repr"] = light_repr[...,:3]
        data["local"]["lightdir_feature"] = self.light_dir_linear(data["local"]["lightdir_feature"]* weights)
        data["local"]["halfvec_feature"] = self.half_vec_linear(data["local"]["halfvec_feature"]* weights)
        data["local"]["direct_light_reprs"] = light_repr
        result = {}
        need_direct, need_shadow, need_indirect = (True, True, True)
        if train_setting != None:
            need_autoencoder, need_direct, need_shadow, need_indirect = (
                train_setting["need_autoencoder"], train_setting["need_direct"], train_setting["need_shadow"],
                train_setting["need_indirect"])
        global_timing_profile = {}
        localData = data["local"]
        ##data, timing_indirect = self.negl_indirect(data, enable_timing_profile)
        if enable_timing_profile:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
        direct_result, shading_clue_timing_profile = self.shading_clue_encoders(localData,
                                                                                enable_timing_profile)
        if enable_timing_profile:
            starter.record()
        result.update(direct_result)

        result["direct_no_shading"] = result["direct_shading"] * data["global"]["max_scale"]
        result["direct_shading"] = result["direct_no_shading"] * result["shadow"]
        
        localData["direct_no_shading"] = localData["direct_shading"] * data["global"]["max_scale"]
        localData["direct_shading"] = localData["direct_shading"] * data["global"]["max_scale"] * localData["shadow"]
        #print_image_exr(localData["direct_shading"] ,"direct_shading_origin")
        result["shading"] = result["direct_shading"] + localData["indirect_shading"]
        localData["shading"] = localData["direct_shading"] + localData["indirect_shading"]
        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)
        # for key in loss_map:
        #     #print("{} loss : ".format(key),loss_map[key].mean())
        return data, result, loss_map, global_timing_profile



class ConeDecoder(nn.Module):
    def __init__(self, configs, loss_config):
        super().__init__()
        self.roughness_table = nn.Parameter(get_roughness_angular_table().unsqueeze(dim=-1))
        self.configs = configs
        self.is_cut = configs["light_cut"]
        self.direction_size = 128
        self.transformer_layer_cnt = 6
        self.embedding_size = 16
        self.stage = configs["stage"]
        self.filter = PartitioningPyramid()
        self.relative = configs["relative"]

        # self.direction_plane_embed = nn.Parameter(
        #     0.01 * torch.randn(3, self.direction_resolution, self.direction_resolution, self.embedding_size),
        #     requires_grad=True)
        self.depth_resolution = 512
        self.angular_size = 64
        self.space_size = 64

        self.loss_config = loss_config
        self.compressed_feature = 32
        self.encoder_way = configs["encoder_way"]
        self.weight_way = configs["weight_way"]
        self.decoder_way = configs["decoder_way"]
        self.specular_feature = configs["specular_dim"]
        self.specular_gbuffer = ["N", "R"]
        self.weight_feature = 16

        self.angular_pool = [1, 2, 4, 8]
        # self.space_pool = [1,4,16,64]
        self.space_pool = [1, 2, 4, 8]
        self.full = configs["full"]
        self.level = configs["light_encoder_level"]
        self.training_level = configs["light_encoder_level"]
        if configs["triplane"]:
            self.triplane_encoder = SimpleDiTransformerPlane(configs,soft_max=True)
        if self.weight_way == "attention":
            self.key_encoder = nn.Sequential(
                fc_layer(8, self.specular_feature),
                fc_layer(self.specular_feature, self.specular_feature),
                fc_layer(self.specular_feature, self.specular_feature)
            )
            self.attention_sample_4ds = CrossAttentionBlock(inner_dim=self.specular_feature,
                                                            cond_dim=self.specular_feature,
                                                            num_heads=16, eps=1e-6)
        elif self.weight_way == "scale_kernel":
            self.kernel_predict_layer = nn.Sequential(
                fc_layer(8, 64),
                fc_layer(64, 64),
                fc_layer(64, 64),
                fc_layer(64, 5 + 64)
            )
        else:
            self.kernel_predict_layer = nn.Sequential(
                fc_layer(8, 64),
                fc_layer(64, 64),
                fc_layer(64, 64),
                fc_layer(64, self.training_level + 1)
            )
        self.loss_func = Loss(loss_config) if loss_config else None

        self.light_size_list = [16, 8, 4]

        specular_feature = self.specular_feature

        self.gbuffer_attention_encoder = nn.Sequential(
            fc_layer(13, self.specular_feature),
            fc_layer(self.specular_feature, self.specular_feature),
            fc_layer(self.specular_feature, self.specular_feature),
            fc_layer(self.specular_feature, self.specular_feature)
        )

        self.channel_cut = False
        self.geo_radiance_cut = True
        if self.channel_cut == True:
            self.light_encoder = NewFourdEncoderLN(input_dim=8,
                                                   feature_list=[specular_feature, int(specular_feature * 1.5),
                                                                 int(specular_feature * 2.25),
                                                                 int(specular_feature * 3.75),
                                                                 int(specular_feature * 5.25),
                                                                 int(specular_feature * 7.5)], level=self.level,
                                                   output_dim=self.specular_feature)
            self.B = 3
        else:
            if False:
                self.light_encoder = NewFourdEncoderLN(input_dim=9,
                                                       feature_list=[specular_feature, int(specular_feature * 1.5),
                                                                     int(specular_feature * 2.25),
                                                                     int(specular_feature * 3.75),
                                                                     int(specular_feature * 5.25),
                                                                     int(specular_feature * 7.5)], level=self.level,
                                                       output_dim=self.specular_feature)
            else:

                self.single_light_encoder_list = nn.ModuleList([])
                self.geo_light_encoder_list = nn.ModuleList([])
                self.output_layer_list = nn.ModuleList([])
                self.compressed_layer_list = nn.ModuleList([])
                if self.full:
                    for i in range(4):
                        self.single_light_encoder_list.append(nn.ModuleList([]))
                        for j in range(4):
                            self.single_light_encoder_list[i].append(
                                NewFourdEncoderLN(input_dim=9,
                                                  feature_list=[specular_feature, int(specular_feature * 1.5),
                                                                ], level=1,
                                                  output_dim=self.specular_feature))

                else:
                    if self.encoder_way == "cnn":
                        for i in range(4):
                            self.single_light_encoder_list.append(
                                NewFourdEncoderLN(input_dim=9,
                                                  feature_list=[specular_feature, int(specular_feature * 1.5),
                                                                ], level=1,
                                                  output_dim=self.specular_feature))
                        print("relu encoder ///////")

                    elif self.encoder_way == "unet":
                        self.single_light_encoder = FourDEncoder(input_dim=9, feature_list=[32, 48, 64, 96], level=4,
                                                                 angular_down_list=[2, 2, 2], space_down_list=[4, 2, 2])
                    elif self.encoder_way == "attention":
                        for i in range(4):
                            self.single_light_encoder_list.append(
                                FourDTransformer(angular_size=self.angular_size // self.angular_pool[i],
                                                 space_size=self.space_size // self.space_pool[i],
                                                 block_size=4,
                                                 num_layers=configs["attention_layer_cnt"],
                                                 num_heads=16,
                                                 input_dim=9,
                                                 hidden_dim=self.specular_feature,
                                                 mlp_dim=self.specular_feature,
                                                 output_dim=self.specular_feature
                                                 ))
                        if configs["triplane"]:
                            self.triplane_light_encoder = FourDTransformer(
                                angular_size=self.angular_size // self.angular_pool[1],
                                space_size=self.space_size // self.space_pool[1],
                                block_size=4,
                                num_layers=configs["attention_layer_cnt"],
                                num_heads=16,
                                input_dim=9,
                                hidden_dim=self.specular_feature * 2,
                                mlp_dim=self.specular_feature * 2,
                                output_dim=self.specular_feature * 2
                            )
                    elif self.encoder_way == "mlp":
                        self.single_light_encoder_list.append(nn.Sequential(fc_layer(9, self.specular_feature),
                                                                            fc_layer(self.specular_feature,
                                                                                     self.specular_feature),
                                                                            fc_layer(self.specular_feature,
                                                                                     self.specular_feature),
                                                                            fc_layer(self.specular_feature,
                                                                                     self.specular_feature)
                                                                            ))
                    if self.weight_way == "true_kernel":

                        for i in range(4):
                            self.output_layer_list.append(
                                fc_layer(self.specular_feature * 16, self.specular_feature * 2))

            self.B = 1

        self.weight_conv = ConvUNetForLight(9 + self.specular_feature, [25 + 4, 25, 25, 25],
                                            dims_and_depths=[
                                                (36, 36),
                                                (48, 48),
                                                (64, 64),
                                                (76, 76),
                                                (96, 96, 76)
                                            ]
                                            )
        self.filter = OnlyKernelPyramid()
        self.decoder = nn.Sequential(
            fc_layer(self.specular_feature * 1 + 14, self.specular_feature * 1),
            fc_layer(self.specular_feature * 1, self.specular_feature * 1),
            fc_layer(self.specular_feature * 1, self.specular_feature * 1),
            fc_layer(self.specular_feature * 1, 1))

    def generate_view_uv(self, data, camera_position, second_lposition, camera_uv):

        sampled_screen_camera = torch.nn.functional.grid_sample(camera_position.permute(0, 3, 1, 2),
                                                                camera_uv.expand(1, self.screen_size, self.screen_size,
                                                                                 2),
                                                                padding_mode="border")
        sampled_screen_camera = sampled_screen_camera.permute(0, 2, 3, 1)
        data["local"]["sampled_screen_camera"] = sampled_screen_camera
        dir = normalize(second_lposition - sampled_screen_camera)
        ####pyexr.write("../testData/sampled_camera.exr",sampled_screen_camera[0].cpu().numpy())
        ####pyexr.write("../testData/camera_uv.exr",camera_uv[0].cpu().numpy())
        ####pyexr.write("../testData/camera_position.exr",camera_position[0].cpu().numpy())
        reflect_forward, reflect_up, reflect_right = get_forward_up_right_tensor(-sampled_screen_camera)
        reflect_up = -reflect_up
        # compensate_data["reflect_mask"] = mask
        # reflect_dir[mask] = 0
        uvw = rotate(reflect_right, reflect_up, reflect_forward, dir)
        standard_dir = normalize(uvw)
        uv = concentric_mapping_hemisphere_3D_to_2D(standard_dir)
        ## perspective back reproject
        # d = math.sqrt(1/3)
        # z = d / standard_dir[...,2:3]
        # uvz = standard_dir * z
        # view_uv = uvz[...,:2] /d
        return uv

    def generate_view_uv_by_dir(self, direction):

        ####pyexr.write("../testData/sampled_camera.exr",sampled_screen_camera[0].cpu().numpy())
        ####pyexr.write("../testData/camera_uv.exr",camera_uv[0].cpu().numpy())
        ####pyexr.write("../testData/camera_position.exr",camera_position[0].cpu().numpy())
        reflect_forward, reflect_up, reflect_right = get_forward_up_right_tensor(direction)
        reflect_up = -reflect_up
        # compensate_data["reflect_mask"] = mask
        # reflect_dir[mask] = 0
        uvw = rotate(reflect_right, reflect_up, reflect_forward, dir)
        standard_dir = normalize(uvw)
        uv = concentric_mapping_hemisphere_3D_to_2D(standard_dir)
        ## perspective back reproject
        # d = math.sqrt(1/3)
        # z = d / standard_dir[...,2:3]
        # uvz = standard_dir * z
        # view_uv = uvz[...,:2] /d
        return uv

    def positional_sample_4d(self, data, feature_texture):
        size = self.light_size
        B, C, W2, H2 = feature_texture.shape

        uv_lu, uv_ru, uv_ld, uv_rd, uv_weight = oct_transform(data["local"]["space_uv"], size)
        camera_position = data["global"]["camera_position{}x{}".format(size, size)]
        second_lposition = data["local"]["gbuffer"]["second_lposition"]
        view_uv_lu = self.generate_view_uv(data, camera_position, second_lposition, uv_lu) / size
        view_uv_ru = self.generate_view_uv(data, camera_position, second_lposition, uv_ru) / size
        view_uv_ld = self.generate_view_uv(data, camera_position, second_lposition, uv_ld) / size
        view_uv_rd = self.generate_view_uv(data, camera_position, second_lposition, uv_rd) / size
        ####pyexr.write("../testData/uv_ll1.exr", uv_lu[0].cpu().numpy())
        ####pyexr.write("../testData/bias1.exr", view_uv_lu[0].cpu().numpy())
        data["local"]["positional_uv"] = view_uv_lu
        radiance_lu = torch.nn.functional.grid_sample(feature_texture,
                                                      (uv_lu + view_uv_lu).expand(self.B, self.screen_size,
                                                                                  self.screen_size, 2),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_ru = torch.nn.functional.grid_sample(feature_texture,
                                                      (uv_ru + view_uv_ru).expand(self.B, self.screen_size,
                                                                                  self.screen_size, 2),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_lt = torch.nn.functional.grid_sample(feature_texture,
                                                      (uv_ld + view_uv_ld).expand(self.B, self.screen_size,
                                                                                  self.screen_size, 2),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_rr = torch.nn.functional.grid_sample(feature_texture,
                                                      (uv_rd + view_uv_rd).expand(self.B, self.screen_size,
                                                                                  self.screen_size, 2),
                                                      padding_mode="border").permute(0, 2, 3, 1)

        sampled_feature = torch.cat(
            [radiance_lu.unsqueeze(0), radiance_ru.unsqueeze(0), radiance_lt.unsqueeze(0), radiance_rr.unsqueeze(0)],
            dim=0)

        # ###pyexr.write("../testData/Position_sampled_radiance0.exr", radiance_lu.permute(3,1,2,0)[0].cpu().numpy())
        # ###pyexr.write("../testData/Position_sampled_radiance1.exr", radiance_ru.permute(3,1,2,0)[0].cpu().numpy())
        # ###pyexr.write("../testData/Position_sampled_radiance2.exr", radiance_lt.permute(3,1,2,0)[0].cpu().numpy())
        # ###pyexr.write("../testData/Position_sampled_radiance3.exr", radiance_rr.permute(3,1,2,0)[0].cpu().numpy())
        return sampled_feature, uv_weight[0:1, ...]

    def generate_ray_key_by_gbuffer(self, data):
        if self.channel_cut:
            geo_gbuffer_input = torch.cat([data["local"]["gbuffer"]["lposition"], data["local"]["gbuffer"]["normal"],
                                           data["local"]["origin_view_dir"], data["local"]["gbuffer"]["roughness"],
                                           data["local"]["gbuffer"]["second_lposition"]], dim=-1)
            data["local"]["key"] = self.gbuffer_attention_encoder(geo_gbuffer_input).repeat(3, 1, 1, 1)
        else:
            geo_gbuffer_input = torch.cat([data["local"]["gbuffer"]["lposition"], data["local"]["gbuffer"]["normal"],
                                           data["local"]["origin_view_dir"], data["local"]["gbuffer"]["roughness"],
                                           data["local"]["gbuffer"]["second_lposition"]], dim=-1)
            data["local"]["key"] = self.gbuffer_attention_encoder(geo_gbuffer_input)
        return None

    def sample_4d(self, radiance_texture, angular_uv, space_uv):

        B, W1, H1, W2, H2, C = radiance_texture.shape

        geo_size = W1
        radiance_texture = radiance_texture.permute(0, 1, 3, 2, 4, 5).reshape(B, W1 * W2, H1 * H2, C).permute(0, 3, 1,
                                                                                                              2)

        uv_ll, uv_rl, uv_lr, uv_rr, uv_weight = oct_transform(angular_uv, geo_size)
        # ##pyexr.write("../testData/uv_ll0.exr", uv_ll[0].detach().cpu().numpy())
        bias = space_uv / geo_size

        radiance_lu = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_ll + bias),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_ru = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_rl + bias),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_lt = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_lr + bias),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_rr = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_rr + bias),
                                                      padding_mode="border").permute(0, 2, 3, 1)

        sampled_feature = torch.cat(
            [radiance_lu.unsqueeze(0), radiance_ru.unsqueeze(0), radiance_lt.unsqueeze(0), radiance_rr.unsqueeze(0)],
            dim=0)
        uv_weight = uv_weight[0:1]
        sampled_feature_final = None
        for j in range(4):
            if j == 0:
                sampled_feature_final = sampled_feature[j, ...] * uv_weight[..., j:j + 1]
            else:
                sampled_feature_final = sampled_feature_final + sampled_feature[j, ...] * uv_weight[..., j:j + 1]
        return sampled_feature, uv_weight[0:1, ...], sampled_feature_final

    def sample_4d_modify(self, radiance_texture, angular_uv, standard_dir):
        B, W1, H1, W2, H2, C = radiance_texture.shape

        geo_size = W1
        radiance_texture = radiance_texture.permute(0, 1, 3, 2, 4, 5).reshape(B, W1 * W2, H1 * H2, C).permute(0, 3, 1,
                                                                                                              2)

        uv_ll, uv_rl, uv_lr, uv_rr, uv_weight = oct_transform(angular_uv, geo_size)
        camera_position = equal_area_square_to_sphere(torch.cat([uv_ll, uv_rl, uv_lr, uv_rr], dim=0))
        to_centor = -camera_position
        reflect_forward, reflect_up, reflect_right = get_forward_up_right_tensor(to_centor)
        reflect_up = -reflect_up
        uvw = rotate(reflect_right, reflect_up, reflect_forward, standard_dir.repeat(4, 1, 1, 1))
        standard_dir = normalize(uvw)
        space_uv = concentric_mapping_hemisphere_3D_to_2D(standard_dir)
        for i in range(4):
            pyexr.write("../testData/space_uv{}.exr".format(i), space_uv[i].cpu().numpy())
        bias = space_uv / geo_size

        radiance_lu = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_ll + bias[:1, ...]),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_ru = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_rl + bias[1:2, ...]),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_lt = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_lr + bias[2:3, ...]),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_rr = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_rr + bias[3:4, ...]),
                                                      padding_mode="border").permute(0, 2, 3, 1)

        sampled_feature = torch.cat(
            [radiance_lu.unsqueeze(0), radiance_ru.unsqueeze(0), radiance_lt.unsqueeze(0), radiance_rr.unsqueeze(0)],
            dim=0)
        uv_weight = uv_weight[0:1]
        sampled_feature_final = None
        for j in range(4):
            if j == 0:
                sampled_feature_final = sampled_feature[j, ...] * uv_weight[..., j:j + 1]
            else:
                sampled_feature_final = sampled_feature_final + sampled_feature[j, ...] * uv_weight[..., j:j + 1]
        return sampled_feature, uv_weight[0:1, ...], sampled_feature_final

    def sample_4d_attention(self, radiance_texture, angular_uv, space_uv, key):

        B, W1, H1, W2, H2, C = radiance_texture.shape

        geo_size = W1
        radiance_texture = radiance_texture.permute(0, 1, 3, 2, 4, 5).reshape(B, W1 * W2, H1 * H2, C).permute(0, 3, 1,
                                                                                                              2)

        uv_ll, uv_rl, uv_lr, uv_rr, uv_weight = oct_transform(angular_uv, geo_size)
        # ##pyexr.write("../testData/uv_ll0.exr", uv_ll[0].detach().cpu().numpy())
        bias = space_uv / geo_size

        radiance_lu = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_ll + bias),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_ru = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_rl + bias),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_lt = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_lr + bias),
                                                      padding_mode="border").permute(0, 2, 3, 1)
        radiance_rr = torch.nn.functional.grid_sample(radiance_texture,
                                                      (uv_rr + bias),
                                                      padding_mode="border").permute(0, 2, 3, 1)

        sampled_feature = torch.cat(
            [radiance_lu.unsqueeze(0), radiance_ru.unsqueeze(0), radiance_lt.unsqueeze(0), radiance_rr.unsqueeze(0)],
            dim=0).permute(1, 2, 0, 3)

        uv_weight = uv_weight[0:1]
        sampled_feature_final = None
        for j in range(4):
            if j == 0:
                sampled_feature_final = sampled_feature[j, ...] * uv_weight[..., j:j + 1]
            else:
                sampled_feature_final = sampled_feature_final + sampled_feature[j, ...] * uv_weight[..., j:j + 1]
        return sampled_feature, uv_weight[0:1, ...], sampled_feature_final

    def direction_padding(self, radiance_texture, padding_size=1):
        B, W1, H1, W2, H2, C = radiance_texture.shape
        radiance_texture = radiance_texture.reshape(B * W1 * H1, W2, H2, C).permute(0, 3, 1, 2)
        W2 = W2 + padding_size * 2
        H2 = H2 + padding_size * 2
        return F.pad(radiance_texture, (padding_size, padding_size, padding_size, padding_size),
                     mode="replicate").permute(0, 2, 3, 1).reshape(B, W1, H1, W2, H2, C)

    def sample_4d_kernel(self, data, radiance_texture, sample_kernel, light_cut=None):
        B, W1, H1, W2, H2, C = radiance_texture.shape

        # print(radiance_texture.shape)
        # if C == 1:
        # ##pyexr.write("../testData/radiance_texture2.exr",radiance_texture.permute(1,3,2,4,5,0).reshape(W1*W2,H1*H2,-1).detach().cpu().numpy())
        radiance_texture = self.direction_padding(radiance_texture, 2)

        W2_2 = W2 + 4
        H2_2 = H2 + 4
        space_uv = data["local"]["space_uv"]
        # if light_cut != None:
        #     self.is_cut = light_cut
        # if  self.is_cut:
        #     space_uv = data["local"]["space_uv_co"]

        uv_ll, uv_rl, uv_lr, uv_rr, uv_weight = oct_transform(space_uv, W1)
        # ##pyexr.write("../testData/uv_ll2.exr",uv_ll[0].detach().cpu().numpy())
        kernel_bias = ((data["local"]["spherical_uv"] + 1) / 2 * W2 + 0.5) % 1
        sample_kernel = self.splat(kernel_bias, sample_kernel)
        int_spherical_uv = ((data["local"]["spherical_uv"] + 1) / 2 * W2 - 1.5) // 1 + 2  ## padding + 2
        direction_texture = radiance_texture.reshape(B * W1 * H1, W2_2, H2_2, C).permute(0, 3, 1, 2)
        unfold_radiance_texture = F.unfold(direction_texture, 4).reshape(B * W1 * H1, C * 16, W2 + 1, H2 + 1)

        _, C16, W2, H2 = unfold_radiance_texture.shape
        float_spherical_uv = ((int_spherical_uv / (W2_2 - 3)) * 2 - 1) / W1
        radiance_texture = unfold_radiance_texture.reshape(B, W1, H1, C16, W2, H2).permute(0, 3, 1, 4, 2, 5).reshape(B,
                                                                                                                     C16,
                                                                                                                     W1 * W2,
                                                                                                                     H1 * H2)

        bias = float_spherical_uv

        # gg = (uv_ll + bias).expand(1, self.screen_size, self.screen_size, 2)
        # ###pyexr.write("../testData/uv_ll0.exr", uv_ll[0].cpu().numpy())
        # ###pyexr.write("../testData/bias0.exr", bias[0].cpu().numpy())
        radiance_list = []
        radiance_list.append(torch.nn.functional.grid_sample(radiance_texture,
                                                             (uv_ll + bias).expand(self.B, self.screen_size,
                                                                                   self.screen_size, 2),
                                                             padding_mode="border", mode="nearest").permute(0, 2, 3, 1))
        radiance_list.append(torch.nn.functional.grid_sample(radiance_texture,
                                                             (uv_rl + bias).expand(self.B, self.screen_size,
                                                                                   self.screen_size, 2),
                                                             padding_mode="border", mode="nearest").permute(0, 2, 3, 1))
        radiance_list.append(torch.nn.functional.grid_sample(radiance_texture,
                                                             (uv_lr + bias).expand(self.B, self.screen_size,
                                                                                   self.screen_size, 2),
                                                             padding_mode="border", mode="nearest").permute(0, 2, 3, 1))
        radiance_list.append(torch.nn.functional.grid_sample(radiance_texture,
                                                             (uv_rr + bias).expand(self.B, self.screen_size,
                                                                                   self.screen_size, 2),
                                                             padding_mode="border", mode="nearest").permute(0, 2, 3, 1))
        B, SW, SH, _ = sample_kernel.shape
        ff = sample_kernel.sum(dim=-1)

        for i in range(4):
            # #print(radiance_list[i].max())
            # #print(radiance_list[i].min())
            gg = sample_kernel.unsqueeze(-2)
            # #print(gg.max())
            # #print(gg.min())
            radiance_list[i] = (radiance_list[i].reshape(self.B, SW, SH, C, 16) * sample_kernel.unsqueeze(-2)).sum(
                dim=-1).unsqueeze(0)
            # #print(radiance_list[i].min())
            # #print(radiance_list[i].max())
        sampled_feature = torch.cat(
            radiance_list,
            dim=0)

        return sampled_feature, uv_weight[0:1, ...]

    def generate_dir_light_feature(self, data, light_feature_list):
        light_feature = None
        for i in range(self.level):
            if i == 0:
                light_feature = self.sample_4d(data, light_feature_list[i])
            else:
                sampled_feature = self.sample_4d(data, light_feature_list[i])
                light_feature = torch.cat([light_feature, sampled_feature], dim=-1)
        light_feature = light_feature.reshape(1, self.screen_size, self.screen_size, -1)

        return light_feature

    def generate_attention_light_feature(self, data, light_feature_list, query):
        light_feature = None
        for i in range(self.level):
            if i == 0:
                light_feature = self.attention_sample_4d(data, light_feature_list[i], query)
                # data["local"]["sampled_geo_feature_visualize"] = sampled_geo_feature.reshape(1,512,512,16)[...,:3]
            else:
                sampled_feature = self.attention_sample_4d(data, light_feature_list[i], query)
                light_feature = torch.cat([light_feature, sampled_feature], dim=-1)
        light_feature = light_feature.reshape(3, 512, 512, -1)

        return light_feature

    def attention_sample_4d(self, data, feature_texture, query):
        query = query.reshape(self.B * self.screen_size * self.screen_size, 1, self.specular_feature)
        light_feature, linear_weight = self.positional_sample_4d(data, feature_texture)
        light_feature = light_feature.permute(1, 2, 3, 0, 4).reshape(3 * self.screen_size * self.screen_size, 4, -1)
        attention_light_feature = self.attention_sample_4ds(query, light_feature)
        return attention_light_feature

    def geo_visualize(self, data):
        light_space_geo = self.geo_feature_list[0][:, :3, ...].permute(0, 2, 3, 1)
        data["local"]["light_geo_feature_visualize"] = light_space_geo
        # print("light_sapce_geo_visual shape", light_space_geo.shape)
        return None

    def generate_weight(self, data):
        weight_input = torch.cat([data["local"]["gbuffer"]["lposition"], data["local"]["gbuffer"]["normal"],
                                  data["local"]["gbuffer"]["roughness"]], dim=-1)
        weight = self.weight_network(weight_input)
        weight = torch.softmax(weight, dim=-1)
        data["local"]["weight"] = weight
        return None

    def splat(self, uv_bias, kernel):
        splat_kernel = torch.cat([(1 - uv_bias[..., :1]) * (1 - uv_bias[..., 1:2]),
                                  (uv_bias[..., :1]) * (1 - uv_bias[..., 1:2]),
                                  (1 - uv_bias[..., :1]) * (uv_bias[..., 1:2]),
                                  (uv_bias[..., :1]) * (uv_bias[..., 1:2])], dim=-1)
        B, W, H, C = kernel.shape
        texture = kernel.unsqueeze(-1) * splat_kernel.unsqueeze(-2)
        texture = texture.reshape(B * W * H, 1, 3, 3, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B * W * H, 1, 6, 6)
        texture = torch.nn.functional.avg_pool2d(texture, kernel_size=2, stride=2, padding=1) * 4
        texture = texture.reshape(B, W, H, 16)
        return texture

    def space_padding(self, texture):
        B, W1, H1, W2, H2, C = texture.shape
        result_texture = torch.zeros(B, W1 + 4, H1 + 4, W2, H2, C).cuda()
        result_texture[:, 2:-2, 2:-2, ...] = texture

        for i in range(2):
            for j in range(W1):
                result_texture[:, i, j + 2, ...] = texture[:, i, H1 - j - 1, ...]
                result_texture[:, W1 + 2 + i, j + 2, ...] = texture[:, W1 - i - 1, H1 - j - 1, ...]
                result_texture[:, j + 2, i, ...] = texture[:, W1 - j - 1, i, ...]
                result_texture[:, j + 2, H1 + 2 + i, ...] = texture[:, W1 - j - 1, H1 - i - 1, ...]
        for i in range(2):
            for j in range(2):
                result_texture[:, i, j, ...] = texture[:, W1 - 1 - i, H1 - 1 - j, ...]
                result_texture[:, W1 + 2 + i, j, ...] = texture[:, i, H1 - 1 - j, ...]
                result_texture[:, i, H1 + 2 + j, ...] = texture[:, W1 - 1 - i, j, ...]
                result_texture[:, W1 + 2 + i, H1 + 2 + j, ...] = texture[:, i, j, ...]
        return result_texture

    def forward_light_encoder(self, data):
        result_feature_list = []
        data = data["global"]
        B, W1, H1, W2, H2, C = data["radiance"].shape
        scale = 1
        for i in range(self.level):
            result_feature_list.append(
                torch.zeros(self.B, W1 // scale, H1 // scale, W2 // scale, H2 // scale, self.specular_feature).cuda())
            scale = scale * 2

        cut_resolution = 4
        key_list = ["radiance", "position", "direction", "depth"]
        padding_light_data = {}
        for key in data:
            if key not in key_list:
                continue
            padding_light_data[key] = self.space_padding(data[key])
        # ##pyexr.write("../testData/padding_light_data.exr",padding_light_data["radiance"][0].permute(0,2,1,3,4).reshape(36*128,36 * 128,3).cpu().numpy())
        for i in range(W1 // cut_resolution):
            for j in range(H1 // cut_resolution):
                light_block_data = {}
                for key in data:
                    if key not in key_list:
                        continue
                    light_block_data[key] = padding_light_data[key][:,
                                            i * cut_resolution:((i + 1) * cut_resolution + 4),
                                            j * cut_resolution:((j + 1) * cut_resolution + 4), ...]
                if self.channel_cut:
                    light_input = torch.cat(
                        [light_block_data["radiance"].permute(5, 1, 2, 3, 4, 0),
                         light_block_data["position"].repeat(3, 1, 1, 1, 1, 1),
                         light_block_data["direction"].repeat(3, 1, 1, 1, 1, 1),
                         light_block_data["depth"].repeat(3, 1, 1, 1, 1, 1)],
                        dim=-1)
                else:
                    light_input = torch.cat(
                        [light_block_data["radiance"],
                         light_block_data["position"],
                         light_block_data["direction"],
                         light_block_data["depth"]],
                        dim=-1)
                feature_result_list = self.light_encoder(light_input)
                value = feature_result_list[0][:, 2:-2,
                        2:-2, ...]
                result_feature_list[0][:, i * cut_resolution:(i + 1) * cut_resolution,
                j * cut_resolution:(j + 1) * cut_resolution, ...] = value
        # print(result_feature_list[0].shape)
        return result_feature_list

    def forward_light_encoder2(self, data):
        lightData = data["global"]
        if self.channel_cut:
            light_input = torch.cat(
                [lightData["radiance"].permute(5, 1, 2, 3, 4, 0), lightData["position"].repeat(3, 1, 1, 1, 1, 1),
                 lightData["direction"].repeat(3, 1, 1, 1, 1, 1), lightData["depth"].repeat(3, 1, 1, 1, 1, 1)], dim=-1)
        else:
            light_input = torch.cat(
                [lightData["radiance"], lightData["position"],
                 lightData["direction"], lightData["depth"]], dim=-1)
        result_feature_list = self.light_encoder(light_input)
        return result_feature_list

    def down_dir_light_data(self, data, size):
        pool_f = nn.AvgPool2d(size)
        for key in data["global"]:
            if key in ["radiance", "direction", "depth", "position"]:
                B, W1, H1, W2, H2, C = data["global"][key].shape
                data["global"][key] = pool_f(
                    data["global"][key].reshape(B * W1 * H1, W2, H2, C).permute(0, 3, 1, 2)).permute(0, 2, 3,
                                                                                                     1).reshape(B, W1,
                                                                                                                H1,
                                                                                                                W2 // size,
                                                                                                                H2 // size,
                                                                                                                C)
                print(data["global"][key].shape)

    def sample_4d_kernel_nounfold(self, data, radiance_texture, sample_kernel, light_cut=None):
        B, W1, H1, W2, H2, C = radiance_texture.shape
        B, W1, H1, W2, H2, C = radiance_texture.shape

        print("samplec : ", radiance_texture.shape)
        # if C == 1:
        # ##pyexr.write("../testData/radiance_texture2.exr",radiance_texture.permute(1,3,2,4,5,0).reshape(W1*W2,H1*H2,-1).detach().cpu().numpy())
        radiance_texture = self.direction_padding(radiance_texture, 2)

        W2_2 = W2 + 4
        H2_2 = H2 + 4
        space_uv = data["local"]["space_uv"]
        # if light_cut != None:
        #     self.is_cut = light_cut
        # if  self.is_cut:
        #     space_uv = data["local"]["space_uv_co"]

        uv_ll, uv_rl, uv_lr, uv_rr, uv_weight = oct_transform(space_uv, W1)
        # ##pyexr.write("../testData/uv_ll2.exr",uv_ll[0].detach().cpu().numpy())
        kernel_bias = ((data["local"]["spherical_uv"] + 1) / 2 * W2 + 0.5) % 1
        sample_kernel = self.splat(kernel_bias, sample_kernel)
        radiance_texture = radiance_texture.permute(0, 5, 1, 3, 2, 4).reshape(B, C, W1 * W2_2, H1 * H2_2)
        radiance_list = []
        for i in range(4):
            radiance_list.append([])
        for i in range(16):
            u = i % 4
            v = i - u * 4

            biasuv = torch.Tensor([u, v]).reshape(1, 1, 1, 2).cuda()
            # print(biasuv.shape)
            # print(data["local"]["spherical_uv"].shape)
            int_spherical_uv = ((data["local"]["spherical_uv"] + 1) / 2 * W2 - 1.5) // 1 + 2 + biasuv  ## padding + 2
            bias = ((int_spherical_uv / (W2_2)) * 2 - 1) / W1
            radiance_list[0].append(torch.nn.functional.grid_sample(radiance_texture,
                                                                    (uv_ll + bias).expand(self.B, self.screen_size,
                                                                                          self.screen_size, 2),
                                                                    padding_mode="border", mode="nearest").permute(0, 2,
                                                                                                                   3,
                                                                                                                   1))
            radiance_list[1].append(torch.nn.functional.grid_sample(radiance_texture,
                                                                    (uv_rl + bias).expand(self.B, self.screen_size,
                                                                                          self.screen_size, 2),
                                                                    padding_mode="border", mode="nearest").permute(0, 2,
                                                                                                                   3,
                                                                                                                   1))
            radiance_list[2].append(torch.nn.functional.grid_sample(radiance_texture,
                                                                    (uv_lr + bias).expand(self.B, self.screen_size,
                                                                                          self.screen_size, 2),
                                                                    padding_mode="border", mode="nearest").permute(0, 2,
                                                                                                                   3,
                                                                                                                   1))
            radiance_list[3].append(torch.nn.functional.grid_sample(radiance_texture,
                                                                    (uv_rr + bias).expand(self.B, self.screen_size,
                                                                                          self.screen_size, 2),
                                                                    padding_mode="border", mode="nearest").permute(0, 2,
                                                                                                                   3,
                                                                                                                   1))

        B, SW, SH, _ = sample_kernel.shape

        for i in range(4):
            # print(torch.cat(radiance_list[i],dim=-1).shape)
            radiance_list[i] = (
                    torch.cat(radiance_list[i], dim=-1).reshape(self.B, SW, SH, C, 16) * sample_kernel.unsqueeze(
                -2)).sum(
                dim=-1).unsqueeze(0)

        sampled_feature = torch.cat(
            radiance_list,
            dim=0)

        return sampled_feature, uv_weight[0:1, ...]

    def lod_sampling(self, light_feature_list, lod_local_data):
        C = light_feature_list[0].shape[-1]
        layer_angular_range = [0, 18, 54, 108, 180]
        layer_space_range = [0, 18, 54, 108, 180]
        B, W, H, _ = lod_local_data["angular_range"].shape
        angular_range = lod_local_data["angular_range"].reshape(-1)
        space_range = lod_local_data["space_range"].reshape(-1)
        angular_uv = lod_local_data["angular_uv"].reshape(-1, 2)
        space_uv = lod_local_data["space_uv"].reshape(-1, 2)
        ray_mask = lod_local_data["ray_mask"].reshape(-1)
        # #pyexr.write("../testData/angular_range.exr",lod_local_data["angular_range"][0].cpu().numpy())
        # #pyexr.write("../testData/space_range.exr",lod_local_data["space_range"][0].cpu().numpy())
        # #pyexr.write("../testData/mask_range.exr",lod_local_data["ray_mask"][0].cpu().numpy())
        final_feature = torch.zeros(B * W * H, C).cuda()

        for i in range(self.training_level):  # space
            for j in range(self.training_level):  # angular
                angular_range_bottom = torch.pi * layer_angular_range[j] / 180
                angular_range_up = torch.pi * layer_angular_range[j + 1] / 180
                space_range_bottom = torch.pi * layer_space_range[i] / 180
                space_range_up = torch.pi * layer_space_range[i + 1] / 180
                mask = ((angular_range >= (angular_range_bottom - 1e-5)) & (angular_range < angular_range_up) \
                        & (space_range >= (space_range_bottom - 1e-5)) & (space_range < space_range_up) & ray_mask)
                # #pyexr.write("../testData/mask{}_{}_0.exr".format(i,j),mask.reshape(4,512,512,1)[0].cpu().numpy())
                # #pyexr.write("../testData/mask{}_{}_1.exr".format(i,j),mask.reshape(4,512,512,1)[1].cpu().numpy())
                # #pyexr.write("../testData/mask{}_{}_2.exr".format(i,j),mask.reshape(4,512,512,1)[2].cpu().numpy())
                # #pyexr.write("../testData/mask{}_{}_3.exr".format(i,j),mask.reshape(4,512,512,1)[3].cpu().numpy())
                gg = torch.where(mask)
                indices = gg[0]
                if indices.shape[0] == 0:
                    continue
                local_space_uv = space_uv[indices].unsqueeze(0).unsqueeze(0)
                local_angular_uv = angular_uv[indices].unsqueeze(0).unsqueeze(0)
                __, _, feature = self.sample_4d(light_feature_list[i * 4 + j], local_angular_uv, local_space_uv)
                # __, _, feature = self.sample_4d(light_feature_list[i * 4 + j], local_angular_uv, local_space_uv)
                final_feature[indices] = feature.reshape(-1, C)

        final_feature = final_feature.reshape(B, W, H, C)
        return final_feature

    def lod_sampling_simple(self, light_feature_list, lod_local_data):
        C = light_feature_list[0].shape[-1]
        layer_angular_range = [0, 18, 54, 108, 180]
        layer_space_range = [0, 18, 54, 108, 180]
        B, W, H, _ = lod_local_data["angular_range"].shape
        angular_uv = lod_local_data["angular_uv"].reshape(4, -1, 2)
        space_uv = lod_local_data["space_uv"].reshape(4, -1, 2)
        ray_mask = lod_local_data["ray_mask"].reshape(4, -1)
        shuaijian = lod_local_data["shuaijian"].reshape(4, -1)
        # #pyexr.write("../testData/angular_range.exr",lod_local_data["angular_range"][0].cpu().numpy())
        # #pyexr.write("../testData/space_range.exr",lod_local_data["space_range"][0].cpu().numpy())
        # #pyexr.write("../testData/mask_range.exr",lod_local_data["ray_mask"][0].cpu().numpy())
        final_feature = torch.zeros(4, W * H, C).cuda()
        # print("training_level",self.training_level)
        for i in range(self.training_level):  # space
            a_uv = angular_uv[i, ...]
            s_uv = space_uv[i, ...]
            mask = ray_mask[i]
            # #pyexr.write("../testData/mask{}_{}_0.exr".format(i,j),mask.reshape(4,512,512,1)[0].cpu().numpy())
            # #pyexr.write("../testData/mask{}_{}_1.exr".format(i,j),mask.reshape(4,512,512,1)[1].cpu().numpy())
            # #pyexr.write("../testData/mask{}_{}_2.exr".format(i,j),mask.reshape(4,512,512,1)[2].cpu().numpy())
            # #pyexr.write("../testData/mask{}_{}_3.exr".format(i,j),mask.reshape(4,512,512,1)[3].cpu().numpy())
            gg = torch.where(mask)
            indices = gg[0]
            if indices.shape[0] == 0:
                continue
            a_uv = a_uv[indices].unsqueeze(0).unsqueeze(0)
            single_shuaijian = shuaijian[i, indices].unsqueeze(-1)
            s_uv = s_uv[indices].unsqueeze(0).unsqueeze(0)
            # print("light_feauture max: {}".format(i),light_feature_list[i].max())
            # print("a_uv max: {}".format(i),a_uv.max())
            # print("s_uv max: {}".format(i),s_uv.max())
            # print("single_shuaijian max: {}".format(i),single_shuaijian.max())
            __, _, feature = self.sample_4d(light_feature_list[i], a_uv, s_uv)
            # __, _, feature = self.sample_4d(light_feature_list[i], local_angular_uv, local_space_uv)
            final_feature[i, indices] = feature.reshape(-1, C) * single_shuaijian

        final_feature = final_feature.reshape(B, W, H, C)

        return final_feature

    # def cut_sample_area(self,index,angular_size,space_size): #N,8
    #     ## space area dont need cut
    #     angular_index = torch.cat([index[...,:2],index[...,4:6]],dim=-1)
    #
    #     return cut_index
    def forward(self, data, ligih_feature_list=None):
        self.screen_size = data["local"]["specular_shading"].shape[1]
        # self.down_dir_light_data(data, 2)
        # print("screen_size ",self.screen_size)
        # print(data["global"]["radiance"].shape)

        lightData = data["global"]
        specular_reflect_dir = torch.sum(data["local"]["gbuffer"]["view_dir"] * data["local"]["gbuffer"]["normal"],
                                         dim=-1).unsqueeze(-1) * data["local"]["gbuffer"]["normal"] * 2 - \
                               data["local"]["gbuffer"]["view_dir"]
        data["local"]["gbuffer"]["specular_ray"] = specular_reflect_dir
        data["local"]["gbuffer"]["half_vec"] = normalize(
            (normalize(-data["local"]["gbuffer"]["lposition"]) + data["local"]["gbuffer"]["view_dir"]) / 2)
        data["local"]["gbuffer"]["dot"] = torch.sum(
            data["local"]["gbuffer"]["half_vec"] * data["local"]["gbuffer"]["normal"], dim=-1, keepdim=True)
        lod_local_data = get_lod_tracing(data["local"]["gbuffer"]["lposition"], specular_reflect_dir,
                                         data["local"]["gbuffer"]["normal"])
        ####print_image_exr(data["local"]["specular_shading"],"specular_shading")
        ####print_image_exr(data["local"]["gbuffer"]["normal"],"normal")
        lod_local_data_new = get_lod_tracing_arbitrary(data["local"]["gbuffer"]["lposition"], specular_reflect_dir,
                                                       data["local"]["gbuffer"]["normal"],
                                                       1 - data["local"]["gbuffer"]["roughness"],
                                                       self.roughness_table, self.angular_size, self.space_size)
        light_space_gbuffer = get_light_space_gbuffer(data)

        sample_index = torch.cat([lod_local_data_new["angular_uv_left"], lod_local_data_new["space_uv_left"],
                                  lod_local_data_new["angular_uv_right"], lod_local_data_new["space_uv_right"]], dim=-1)
        sample_index = torch.where(torch.isnan(sample_index), 0, sample_index).long()
        ###print_image_exr(sample_index[...,:4],"four_index")
        ###print_image_exr(sample_index[...,4:],"four_index2")
        # print(lightData["radiance"][0].min())
        # print(lightData["radiance"][0].max())
        # print(compute_4d_prefix_sum(lightData["radiance"][0]).min())
        # print(compute_4d_prefix_sum(lightData["radiance"][0]).max())
        ###print_image_exr(data["local"]["gbuffer"]["roughness"], "true_roughness")
        sampled_value = batch_get_4d_region_sum(compute_4d_prefix_sum(lightData["radiance"][0]),
                                                sample_index[0].reshape(-1, 8)).reshape(1, 512, 512, 3)
        ###print_image_exr(sampled_value,"sampled_specular")
        ###print_image_exr(data["local"]["gbuffer"]["normal"],"normal")
        area = sample_index[..., 4:] - sample_index[..., :4]
        area = area[..., :1] * area[..., 1:2] * area[..., 2:3] * area[..., 3:4]
        ###print_image_exr(sampled_value/area,"sampled_specular_after_pool")
        ###print_image_exr(area,"area")
        area_after_scale = area * lod_local_data_new["cone_scale"]
        area_after_scale2 = area * lod_local_data_new["cone_scale2"]
        # cut_index = self.cut_sample_area(sample_index)
        ###print_image_exr(area_after_scale,"area_after_scale")
        ###print_image_exr(sampled_value/area_after_scale,"sampled_specular_after_pool_scale")
        ###print_image_exr(sampled_value/area_after_scale2,"sampled_specular_after_pool_scale2")
        ###print_image_exr(lod_local_data_new["cone_scale"],"scale")
        _, _, sampled_value2 = self.sample_4d(lightData["radiance"], lod_local_data_new["angular_uv"],
                                              lod_local_data_new["space_uv"])
        ###print_image_exr(data["local"]["specular_shading"],"specular_shading")
        ###print_image_exr(sampled_value2,"sampled_specular2")
        # geo_input = torch.cat([lightData["position"],lightData["depth"], lightData["direction"]], dim=-1)
        # geo_feature = self.geo_encoder(geo_input)
        # geo_representation_list = self.geo_encoder(geo_input)
        # second_lposition_mask = torch.norm(data["local"]["gbuffer"]["second_lposition"], dim=-1).unsqueeze(-1) > 0.66
        # second_lposition_mask = second_lposition_mask.repeat(1, 1, 1, 3)
        # data["local"]["second_lposition_mask"] = second_lposition_mask

        # self.generate_ray_key_by_gbuffer(data)
        z = specular_reflect_dir[..., 2:3]

        if self.channel_cut:
            light_input = torch.cat(
                [lightData["radiance"].permute(5, 1, 2, 3, 4, 0), lightData["position"].repeat(3, 1, 1, 1, 1, 1),
                 lightData["direction"].repeat(3, 1, 1, 1, 1, 1)], dim=-1)
        else:
            light_input = torch.cat(
                [lightData["radiance"], lightData["position"],
                 lightData["direction"]], dim=-1)

        # print("light input ",light_input.max())
        light_input_list = []
        if ligih_feature_list == None:
            # result_feature_list = self.light_encoder(light_input)
            result_feature_list = []
            # base_feature_list = []
            # for i in range(self.level):
            #     texture = self.single_light_encoder_list[i](light_input)[0]
            #     B,W1,H1,W2,H2,C = texture.shape
            #     base_feature_list.append(texture)
            if self.full:
                for i in range(self.level):
                    for j in range(self.level):
                        index = (i + j) // 2
                        texture = light_input

                        B, W1, H1, W2, H2, C = texture.shape
                        space_pool = nn.AvgPool2d(self.space_pool[i])
                        angular_pool = nn.AvgPool2d(self.angular_pool[j])
                        texture = texture.permute(0, 1, 2, 5, 3, 4).reshape(-1, C, W2, H2)
                        texture = space_pool(texture)
                        _, C, W2, H2 = texture.shape
                        texture = texture.reshape(B, W1, H1, C, W2, H2).permute(0, 4, 5, 3, 1, 2).reshape(B * W2 * H2,
                                                                                                          C,
                                                                                                          W1, H1)
                        # print("third :",texture.shape)
                        texture = angular_pool(texture)
                        # print("fourth :",texture.shape)
                        _, C, W1, H1 = texture.shape
                        texture = texture.reshape(B, W2, H2, C, W1, H1).permute(0, 4, 5, 1, 2, 3)
                        texture = self.single_light_encoder_list[i][j](texture)[0]
                        # texture = self.single_light_decoder_list[i](texture)
                        result_feature_list.append(texture)
            else:
                for i in range(4):
                    texture = light_input
                    B, W1, H1, W2, H2, C = texture.shape
                    space_pool = nn.AvgPool2d(self.space_pool[i])
                    angular_pool = nn.AvgPool2d(self.angular_pool[i])
                    texture = texture.permute(0, 1, 2, 5, 3, 4).reshape(-1, C, W2, H2)
                    texture = space_pool(texture)
                    _, C, W2, H2 = texture.shape
                    texture = texture.reshape(B, W1, H1, C, W2, H2).permute(0, 4, 5, 3, 1, 2).reshape(B * W2 * H2, C,
                                                                                                      W1, H1)
                    # print("third :",texture.shape)
                    texture = angular_pool(texture)
                    # print("fourth :",texture.shape)
                    _, C, W1, H1 = texture.shape
                    texture = texture.reshape(B, W2, H2, C, W1, H1).permute(0, 4, 5, 1, 2, 3)
                    light_input_list.append(texture)
                    if i >= self.training_level:
                        break
                    texture = self.single_light_encoder_list[i](texture)
                    C = texture.shape[-1]
                    if self.weight_way == "true_kernel":
                        texture = texture.reshape(B, W1, H1, W2 // 4, 4, H2 // 4, 4, C).permute(0, 1, 2, 3, 5, 4, 6,
                                                                                                7).reshape(B, W1, H1,
                                                                                                           W2 // 4,
                                                                                                           H2 // 4,
                                                                                                           16 * C)
                        texture = self.output_layer_list[i](texture)
                    print(texture.shape)
                    # texture = self.single_light_decoder_list[i](texture)
                    texture = compute_4d_prefix_sum(texture)
                    result_feature_list.append(texture)
                if self.configs["triplane"]:
                    triplane_texture = self.triplane_light_encoder(light_input_list[1])
                    self.triplane_encoder.forward_triplane(triplane_texture)
                    triplane_feature = \
                        self.triplane_encoder.fetch_spherical_tri_plane(data["local"]["gbuffer"]["lposition"])[0]
        else:
            result_feature_list = ligih_feature_list

        if self.full:
            sampled_feature = self.lod_sampling(result_feature_list, lod_local_data)
        else:
            sampled_feature = self.lod_sampling_simple(result_feature_list, lod_local_data)

        # sampled_radiance = self.lod_sampling_simple(result_radiance_list, lod_local_data)
        for i in range(self.training_level):
            # data["local"]["sampled_radiance{}".format(i)] = sampled_radiance[i:i + 1, ...]
            data["local"]["sampled_feature{}".format(i)] = sampled_feature[i:i + 1, ..., :3]

        l_lenth = torch.norm(data["local"]["gbuffer"]["lposition"], dim=-1, keepdim=True)
        l_lenth_mask = l_lenth < 1
        if self.weight_way == "attention":

            key_input = torch.cat([lod_local_data["local_normal"], lod_local_data["local_ray"],
                                   data["local"]["gbuffer"]["roughness"],
                                   l_lenth], dim=-1)

            key = self.key_encoder(key_input)
            final_feature, layer_weight = self.attention_sample_4ds(key.reshape(-1, 1, self.specular_feature),
                                                                    sampled_feature.permute(1, 2, 0, 3).reshape(-1, 4,
                                                                                                                self.specular_feature))
            _, SW, SH, _ = data["local"]["specular_shading"].shape
            final_feature = final_feature.reshape(1, SW, SH, self.specular_feature)
            layer_weight = layer_weight.reshape(1, SW, SH, 4)
        else:

            local_ray_mask = torch.isnan(lod_local_data["local_ray"])
            lod_local_data["local_ray"][local_ray_mask] = 0
            lod_local_data["local_normal"][local_ray_mask] = 0
            key_input = torch.cat([lod_local_data["local_normal"], lod_local_data["local_ray"],
                                   data["local"]["gbuffer"]["roughness"],
                                   l_lenth], dim=-1)

            weight = self.kernel_predict_layer(key_input)
            layer_weight = torch.nn.functional.softmax(weight[..., :5], dim=-1)
            # layer_weight = weight
            B, W, H, C = weight.shape
            feature_c = sampled_feature.shape[-1]
            if self.weight_way == "true_kernel":
                kernel_weight = torch.nn.functional.softmax(weight[..., 5:].reshape(B, W, H, 4, 16), dim=-1)
                for i in range(self.level):
                    if i == 0:
                        final_feature = (sampled_feature[i:i + 1, ...].reshape(1, W, H, 16,
                                                                               feature_c // 16) * kernel_weight[:, :, :,
                                                                                                  i, :].unsqueeze(
                            -1)).reshape(1, W, H, -1) * layer_weight[..., i:i + 1]
                    else:
                        final_feature = final_feature + (
                                sampled_feature[i:i + 1, ...].reshape(1, W, H, 16, feature_c // 16) * kernel_weight[
                                                                                                      :, :, :, i,
                                                                                                      :].unsqueeze(
                            -1)).reshape(1, W, H, -1) * layer_weight[..., i:i + 1]
            if self.weight_way == "layer" or self.stage == 1:

                for i in range(self.level):
                    # print("sampled_feature max {}".format(i),sampled_feature[i:i + 1, ...].max())
                    if i == 0:
                        final_feature = sampled_feature[i:i + 1, ...] * layer_weight[..., i:i + 1]
                    else:
                        final_feature = final_feature + sampled_feature[i:i + 1, ...] * layer_weight[..., i:i + 1]
                if self.configs["triplane"]:
                    final_feature = final_feature + triplane_feature
            # print("layer_weight max ",layer_weight.max())

        data["local"]["kernel0"] = layer_weight[..., 0:4]

        # exit()
        # sampled_position,weight = self.positional_sample_4d(data,position_map.permute(1,0,2,3,))

        sampled_radiance_final = None
        # #print(sampled_radiance.shape)
        # #print(weight.shape)
        # sampled_feature_final = None
        # for i in range(4):
        #     if i == 0:
        #         sampled_feature_final = sampled_feature[i, ...] * weight[..., i:i + 1]
        #     else:
        #         sampled_feature_final = sampled_feature_final + sampled_feature[i, ...] * weight[..., i:i + 1]

        # data["local"]["sample_second_position"] = self.sample_4d(data,self.geo1)

        # print("input max ",specular_input.max())
        data["local"]["local_normal"] = lod_local_data["local_normal"]
        data["local"]["local_ray"] = lod_local_data["local_ray"]
        if self.decoder_way == "pixel_decoder" or self.stage == 1:
            specular_input = torch.cat(
                [final_feature, data["local"]["toLight"][..., :3], data["local"]["gbuffer"]["lposition"],
                 data["local"]["gbuffer"]["normal"],
                 data["local"]["gbuffer"]["view_dir"],
                 data["local"]["gbuffer"]["roughness"],
                 l_lenth], dim=-1)
            specular_shading = self.decoder(specular_input)
            # print("pixel decoder now")
        else:
            specular_input = torch.cat(
                [sampled_feature[:1, ...], data["local"]["toLight"][..., :3], data["local"]["gbuffer"]["normal"],
                 data["local"]["gbuffer"]["lposition"],
                 data["local"]["gbuffer"]["roughness"],
                 data["local"]["gbuffer"]["view_dir"]], dim=-1)
            down_pool = nn.AvgPool2d(2)

            weights = self.weight_conv(specular_input.permute(0, 3, 1, 2), sampled_feature, self.training_level)
            # print(weights[0].shape)
            # print(F.softmax(weights[0][:, 25:25 + self.training_level + 1, ...].permute(0, 2, 3, 1),
            #                                      dim=-1).shape)
            data["local"]["kernel0"] = F.softmax(
                weights[0][:, 25:25 + self.training_level + 1, ...].permute(0, 2, 3, 1),
                dim=-1)[..., :4]
            final_feature = self.filter(weights, sampled_feature.permute(0, 3, 1, 2), self.training_level).permute(0, 2,
                                                                                                                   3, 1)
            specular_input = torch.cat(
                [final_feature, data["local"]["gbuffer"]["normal"], data["local"]["gbuffer"]["lposition"],
                 data["local"]["gbuffer"]["roughness"],
                 data["local"]["gbuffer"]["view_dir"]], dim=-1)
            specular_shading = self.decoder(specular_input)
            print("cnn decoder now")
        if self.channel_cut:
            specular_shading = specular_shading.permute(3, 1, 2, 0)
        localData = data["local"]
        result = {}
        # mask = (data["local"]["gbuffer"]["roughness"] < 0.75).repeat(1,1,1,3)
        result["log1p_specular_shading"] = specular_shading

        result["log1p_specular_shading"] = torch.where(l_lenth_mask,
                                                       localData["log1p_specular_shading"],
                                                       result["log1p_specular_shading"])
        # print("result max ",result["log1p_specular_shading"].max())
        # print("ref max ",localData["log1p_specular_shading"].max())
        if torch.any(torch.isnan(specular_shading)):
            print("nan")
            exit()
        if not self.relative:
            result["specular_shading"] = expp1_torch(result["log1p_specular_shading"]) * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"]

            data["local"]["specular_shading"] = data["local"]["specular_shading"] * data["local"]["gbuffer"][
                "specular"] * \
                                                data["global"]["max_scale"]
        else:
            print("yes forward relative")
            # result["direct_shading"] = result["specular_shading"] + data["local"]["direct_shading"] - data["local"]["specular_shading"]
            result["specular_shading"] = expp1_torch(result["log1p_specular_shading"]) * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"] / data["local"]["relative"]
            data["local"]["specular_shading"] = data["local"]["specular_shading"] * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"] / data["local"]["relative"]

        # result["shadow"] = localData["shadow"]
        # result["log1p_diffuse_shading"] = localData["log1p_diffuse_shading"]
        # result["diffuse_shading"] = expp1_torch(result["log1p_diffuse_shading"])
        # #print("result: ",result["log1p_specular_shading"].min())
        # #print("gt: ",data["local"]["log1p_specular_shading"].min())
        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)
        return data, result, loss_map, {}



class AccConeDecoder(nn.Module):
    def __init__(self, configs, loss_config):
        super().__init__()
        self.roughness_table = nn.Parameter(get_roughness_angular_table().unsqueeze(dim=-1), requires_grad=False)

        self.configs = configs
        self.relative = configs["relative"]
        self.angular_size = 32
        self.space_size = 64
        self.local_gbuffer = configs["local_gbuffer"]
        self.loss_config = loss_config

        self.angular_pool = [1, 2, 4, 8]
        self.space_pool = [1, 2, 4, 8]
        self.channel_cut = True

        if self.channel_cut:
            self.principle_triplane_encoder = ChannelCutDiTransformerPlane(configs)
        else:
            self.principle_triplane_encoder = SimpleDiTransformerPlane(configs)
        self.specular_feature = configs["specular_dim"]
        self.light_input_dim = 9

        if self.channel_cut:
            self.light_input_dim = 7
        self.global_light_encoder = FourDTransformer(
                                                        angular_size=self.angular_size // self.angular_pool[1],
                                                        space_size=self.space_size // self.space_pool[1],
                                                        angular_block_size=2,
                                                        space_block_size=8,
                                                        num_layers=configs["attention_layer_cnt"],
                                                        num_heads=16,
                                                        input_dim=self.light_input_dim,
                                                        hidden_dim=self.specular_feature * 4,
                                                        mlp_dim=self.specular_feature * 1,
                                                        output_dim=self.specular_feature * 2,
                                                        linear_output=False,
                                                     )
        self.specular_feature = configs["plane"]["decoder_direct_dim"]

        self.loss_func = Loss(loss_config) if loss_config else None

        specular_feature = self.specular_feature
        self.output_dim = 3
        self.conditional_layer = False
        if self.channel_cut:
            self.output_dim = 1
        if self.local_gbuffer:
            self.decoder = nn.Sequential(
                fc_layer(self.specular_feature * 1 + 12, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.output_dim))
        else:
            self.decoder = nn.Sequential(
                fc_layer(self.specular_feature * 1 + 15, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.output_dim))
        self.dir_data = self.read_dir(str(self.angular_size) + "x" + str(self.space_size))
        self.dir_data = nn.Parameter(torch.Tensor(self.dir_data["direction"] / \
                                                  numpy.linalg.norm(self.dir_data["direction"], axis=-1)[
                                                      ..., numpy.newaxis]).unsqueeze(0), requires_grad=False)

    def get_lightformer_input(self, localLightData, localData,channel_cnt):
        z_f = localLightData['shadow']['pixel_emitter_distance'] / 3
        z = localLightData['shadow']['occluder_emitter_distance'] / 3

        position_mask = localData["position_mask"]
        z_f[position_mask[..., 0]] = 3
        z[position_mask[..., 0]] = -3

        normal = localData["gbuffer"]["normal"]

        toLight = localData["toLight"][..., :3]
        toView = localData["gbuffer"]["view_dir"]
        c_c = torch.sum(localData["gbuffer"]['normal'] * localData["gbuffer"]['view_dir'], axis=-1)[
            ..., None]  # diffuse # TODO: Use this feature?
        depth = localData["gbuffer"]["depth"]
        localData["z_f_jian_z"] = z_f - z
        localData["z_f_chu_z"] = z_f / z
        localData["normal_dot_view"] = c_c
        localData["hard_shadow"] = torch.where(z_f - z > 8e-4 + localData["tanh_shadow"] * 4e-4, 0, 1)
        return torch.cat([z_f - z, z_f / z, depth[:1,...], c_c[:1,...]], dim=-1)


    def read_dir(self, resolution):
        dir_fn = "../datasets/standard_dir{}.pkl.zst".format(resolution)
        with open(dir_fn, 'rb') as gf:
            dctx = zstd.ZstdDecompressor()
            dir_data = pickle.loads(dctx.decompress(gf.read()))
        gf.close()
        return dir_data

    def fetch_tri_plane(self, triplane, xyz):
        xy = xyz[..., [0, 1]]
        xz = xyz[..., [0, 2]]
        yz = xyz[..., [1, 2]]
        xy_feature = torch.nn.functional.grid_sample(triplane[0:1], xy).permute(0, 2, 3, 1)
        xz_feature = torch.nn.functional.grid_sample(triplane[1:2], xz).permute(0, 2, 3, 1)
        yz_feature = torch.nn.functional.grid_sample(triplane[2:3], yz).permute(0, 2, 3, 1)

        return xy_feature + xz_feature + yz_feature

    def down_sampleing_plane(self, light_input, angular_size, space_size):
        B, W1, H1, W2, H2, C = light_input.shape
        angular_pool = nn.AvgPool2d(angular_size)
        space_pool = nn.AvgPool2d(space_size)
        light_input = space_pool(light_input.permute(0, 1, 2, 5, 3, 4).reshape(B * W1 * H1, C, W2, H2))
        W2, H2 = W2 // space_size, H2 // space_size
        light_input = light_input.reshape(B, W1, H1, C, W2, H2).permute(0, 4, 5, 3, 1, 2).reshape(B * W2 * H2, C, W1,
                                                                                                  H1)
        light_input = angular_pool(light_input)
        W1, H1 = W1 // angular_size, H1 // angular_size
        light_input = light_input.reshape(B, W2, H2, C, W1, H1).permute(0, 4, 5, 1, 2, 3)
        return light_input
    def preprocess_channel_cut(self,data):
        lightData = data["global"]
        lightData["radiance"] = lightData["radiance"].permute(-1, 1, 2, 3, 4, 0)
        lightData["position"] = lightData["position"].repeat(3,1,1,1,1,1)
        lightData["direction"] = lightData["direction"].repeat(3,1,1,1,1,1)
        localData = data["local"]
        for key in localData["gbuffer"]:
            localData["gbuffer"][key] = localData["gbuffer"][key].repeat(3,1,1,1)
        return data
    def forward(self, data, ligih_feature_list=None):
        localData =data["local"]
        SW1, SW2 = data["local"]["specular_shading"].shape[1:3]
        enable_timing_profile = True
        data["global"]["direction"] = self.dir_data
        data = self.preprocess_channel_cut(data)
        lightData = data["global"]
        light_input = torch.cat(
            [lightData["radiance"], lightData["position"],
             lightData["direction"]], dim=-1)

        light_down = self.down_sampleing_plane(light_input, 2, 2)
        global_light_feature = self.global_light_encoder(light_down)
        principle_tri_plane = self.principle_triplane_encoder.forward_triplane(global_light_feature)
        specular_reflect_dir = torch.sum(data["local"]["gbuffer"]["view_dir"] * data["local"]["gbuffer"]["normal"],
                                         dim=-1).unsqueeze(-1) * data["local"]["gbuffer"]["normal"] * 2 - \
                               data["local"]["gbuffer"]["view_dir"]
        data["local"]["gbuffer"]["specular_ray"] = specular_reflect_dir
        data["local"]["gbuffer"]["half_vec"] = normalize(
            (normalize(-data["local"]["gbuffer"]["lposition"]) + data["local"]["gbuffer"]["view_dir"]) / 2)
        data["local"]["gbuffer"]["dot"] = torch.sum(
            data["local"]["gbuffer"]["half_vec"] * data["local"]["gbuffer"]["normal"], dim=-1, keepdim=True)
        W1, W2 = 32, 32
        light_space_gbuffer = get_light_space_gbuffer(data)
        priciple_triplane_index = data["local"]["gbuffer"]["lposition"] / 6
        if not self.channel_cut:
            priciple_triplane_feature = self.fetch_tri_plane(principle_tri_plane, priciple_triplane_index)
        else:
            priciple_triplane_feature,voxel_coord = self.principle_triplane_encoder.fetch_spherical_tri_plane(principle_tri_plane, data["local"]["gbuffer"]["lposition"])
        l_lenth = torch.norm(data["local"]["gbuffer"]["lposition"], dim=-1, keepdim=True)
        data["local"]["sampled_principle_value"] = priciple_triplane_feature[..., :3]
        final_feature = priciple_triplane_feature
        if self.local_gbuffer:
            specular_input = torch.cat(
                [final_feature, light_space_gbuffer["normal"], light_space_gbuffer["half_vec"],
                 light_space_gbuffer["specular_ray"],
                 data["local"]["gbuffer"]["roughness"],
                 data["local"]["gbuffer"]["dot"],
                 l_lenth], dim=-1)
            local_gbuffer_input = torch.cat(
                [light_space_gbuffer["normal"], light_space_gbuffer["half_vec"],
                 light_space_gbuffer["specular_ray"],
                 data["local"]["gbuffer"]["roughness"],
                 data["local"]["gbuffer"]["dot"],
                 l_lenth], dim=-1)
        else:
            specular_input = torch.cat(
                [final_feature, data["local"]["toLight"][..., :3], data["local"]["gbuffer"]["lposition"],
                 data["local"]["gbuffer"]["normal"],
                 data["local"]["gbuffer"]["view_dir"],
                 data["local"]["gbuffer"]["roughness"],
                 data["local"]["gbuffer"]["dot"],
                 l_lenth], dim=-1
            )
        specular_shading = self.decoder(specular_input)
        if self.channel_cut:
            specular_shading = specular_shading.permute(-1,1,2,0)
        result = {}
        result["log1p_specular_shading"] = specular_shading
        result["log1p_specular_shading"] = torch.where(
            data["local"]["mask"] | data["local"]["specular_mask"],
            localData["log1p_specular_shading"],
            result["log1p_specular_shading"])
        localData["pixel_emitter_distance"] = localData["lights"]["shadow"]["pixel_emitter_distance"]
        localData["mask"] = data["local"]["instance_mask"][..., :1] | data["local"]["specular_mask"]
        if torch.any(torch.isnan(result["log1p_specular_shading"])):
            print("nan")
            exit()
        if not self.relative:
            result["specular_shading"] = expp1_torch(result["log1p_specular_shading"]) * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"]

            data["local"]["specular_shading"] = data["local"]["specular_shading"] * data["local"]["gbuffer"][
                "specular"] * \
                                                data["global"]["max_scale"]
        else:
            # result["direct_shading"] = result["specular_shading"] + data["local"]["direct_shading"] - data["local"]["specular_shading"]
            result["specular_shading"] = expp1_torch(result["log1p_specular_shading"]) * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"] / data["local"]["relative"]
            data["local"]["specular_shading"] = data["local"]["specular_shading"] * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"] / data["local"]["relative"]
        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)
        return data, result, loss_map, {}


class ConeDiffuseDecoder(nn.Module):
    def __init__(self, configs, loss_config):
        super().__init__()
        self.configs = configs
        self.filter = PartitioningPyramid()
        self.relative = configs["relative"]
        self.angular_size = 32
        self.space_size = 64

        self.loss_config = loss_config
        self.compressed_feature = 32
        self.specular_feature = configs["specular_dim"]
        self.weight_feature = 16

        if configs["triplane"]:
            self.triplane_encoder = SimpleDiTransformerPlane(configs)
        self.loss_func = Loss(loss_config) if loss_config else None

        if configs["triplane"]:
            self.triplane_light_encoder = FourDTransformer(angular_size=self.angular_size // 2,
                                                           space_size=self.space_size // 2,
                                                           block_size=4,
                                                           num_layers=configs["attention_layer_cnt"],
                                                           num_heads=16,
                                                           input_dim=9,
                                                           hidden_dim=self.specular_feature * 2,
                                                           mlp_dim=self.specular_feature * 2,
                                                           output_dim=self.specular_feature * 2
                                                           )
        self.decoder = nn.Sequential(
            fc_layer(self.specular_feature * 1 + 14, self.specular_feature * 1),
            fc_layer(self.specular_feature * 1, self.specular_feature * 1),
            fc_layer(self.specular_feature * 1, self.specular_feature * 1),
            fc_layer(self.specular_feature * 1, 3))

    def forward(self, data, ligih_feature_list=None):

        lightData = data["global"]

        specular_reflect_dir = torch.sum(data["local"]["gbuffer"]["view_dir"] * data["local"]["gbuffer"]["normal"],
                                         dim=-1).unsqueeze(-1) * data["local"]["gbuffer"]["normal"] * 2 - \
                               data["local"]["gbuffer"]["view_dir"]

        light_input = torch.cat(
            [lightData["radiance"], lightData["position"],
             lightData["direction"]], dim=-1)

        light_input_list = []
        texture = light_input
        B, W1, H1, W2, H2, C = texture.shape
        space_pool = nn.AvgPool2d(2)
        angular_pool = nn.AvgPool2d(2)
        texture = texture.permute(0, 1, 2, 5, 3, 4).reshape(-1, C, W2, H2)
        texture = space_pool(texture)
        _, C, W2, H2 = texture.shape
        texture = texture.reshape(B, W1, H1, C, W2, H2).permute(0, 4, 5, 3, 1, 2).reshape(B * W2 * H2, C,
                                                                                          W1, H1)
        # print("third :",texture.shape)
        texture = angular_pool(texture)
        # print("fourth :",texture.shape)
        _, C, W1, H1 = texture.shape
        texture = texture.reshape(B, W2, H2, C, W1, H1).permute(0, 4, 5, 1, 2, 3)

        if self.configs["triplane"]:
            triplane_texture = self.triplane_light_encoder(texture)
            self.triplane_encoder.forward_triplane(triplane_texture)
            triplane_feature = \
                self.triplane_encoder.fetch_spherical_tri_plane(data["local"]["gbuffer"]["lposition"])[0]

        l_lenth = torch.norm(data["local"]["gbuffer"]["lposition"], dim=-1, keepdim=True)
        l_lenth_mask = l_lenth < 0.6

        specular_input = torch.cat(
            [triplane_feature, data["local"]["toLight"][..., :3], data["local"]["gbuffer"]["lposition"],
             data["local"]["gbuffer"]["normal"],
             data["local"]["gbuffer"]["view_dir"],
             data["local"]["gbuffer"]["roughness"],
             l_lenth], dim=-1)
        specular_shading = self.decoder(specular_input)
        # print("pixel decoder now")

        localData = data["local"]
        result = {}
        mask = data["local"]["pixel_emitter_distance"] == 0
        # mask = (data["local"]["gbuffer"]["roughness"] < 0.75).repeat(1,1,1,3)
        result["log1p_specular_shading"] = specular_shading

        result["log1p_specular_shading"] = torch.where(data["local"]["instance_mask"] | mask,
                                                       localData["log1p_specular_shading"],
                                                       result["log1p_specular_shading"])
        # print("result max ",result["log1p_specular_shading"].max())
        # print("ref max ",localData["log1p_specular_shading"].max())
        if torch.any(torch.isnan(specular_shading)):
            print("nan")
            exit()
        if not self.relative:
            result["specular_shading"] = expp1_torch(result["log1p_specular_shading"]) * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"]

            data["local"]["specular_shading"] = data["local"]["specular_shading"] * data["local"]["gbuffer"][
                "specular"] * \
                                                data["global"]["max_scale"]
        else:
            print("yes forward relative")
            # result["direct_shading"] = result["specular_shading"] + data["local"]["direct_shading"] - data["local"]["specular_shading"]
            result["specular_shading"] = expp1_torch(result["log1p_specular_shading"]) * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"] / data["local"]["relative"]
            data["local"]["specular_shading"] = data["local"]["specular_shading"] * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"] / data["local"]["relative"]

        # result["shadow"] = localData["shadow"]
        # result["log1p_diffuse_shading"] = localData["log1p_diffuse_shading"]
        # result["diffuse_shading"] = expp1_torch(result["log1p_diffuse_shading"])
        # #print("result: ",result["log1p_specular_shading"].min())
        # #print("gt: ",data["local"]["log1p_specular_shading"].min())
        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)
        return data, result, loss_map, {}


class AccShadowDecoder(nn.Module):
    def __init__(self, configs, loss_config):
        super().__init__()
        self.roughness_table = nn.Parameter(get_roughness_angular_table().unsqueeze(dim=-1), requires_grad=False)

        self.configs = configs
        self.relative = configs["relative"]
        self.angular_size = 32
        self.space_size = 64
        self.local_gbuffer = configs["local_gbuffer"]
        self.loss_config = loss_config

        self.angular_pool = [1, 2, 4, 8]
        # self.space_pool = [1,4,16,64]
        self.space_pool = [1, 2, 4, 8]
        self.channel_cut = True

        if self.channel_cut:
            self.principle_triplane_encoder = ChannelCutDiTransformerPlane(configs)
        else:
            self.principle_triplane_encoder = SimpleDiTransformerPlane(configs)
        # self.triplane_encoder = MultiScaleDiTransformerPlane(configs)
        self.specular_feature = configs["specular_dim"]
        self.light_input_dim = 9

        if self.channel_cut:
            self.light_input_dim = 7
        self.global_light_encoder = FourDTransformer(angular_size=self.angular_size // self.angular_pool[1],
                                                     space_size=self.space_size // self.space_pool[1],
                                                     angular_block_size=2,
                                                     space_block_size=8,
                                                     num_layers=configs["attention_layer_cnt"],
                                                     num_heads=16,
                                                     input_dim=self.light_input_dim,
                                                     hidden_dim=self.specular_feature * 4,
                                                     mlp_dim=self.specular_feature * 1,
                                                     output_dim=self.specular_feature * 2,
                                                     linear_output=False
                                                     )
        self.specular_feature = configs["plane"]["decoder_direct_dim"]
        self.loss_func = Loss(loss_config) if loss_config else None

        specular_feature = self.specular_feature

        self.output_dim = 3
        self.conditional_layer = False
        if self.channel_cut:
            self.output_dim = 1
        self.dir_data = self.read_dir(str(self.angular_size) + "x" + str(self.space_size))
        self.dir_data = nn.Parameter(torch.Tensor(self.dir_data["direction"] / \
                                                  numpy.linalg.norm(self.dir_data["direction"], axis=-1)[
                                                      ..., numpy.newaxis]).unsqueeze(0), requires_grad=False)
        self.shadow_network = SmallKernelShadowNetwork(21)
        self.direct_compress_layer = nn.Linear(self.specular_feature, 16)
    def init(self):
        # self.space_forward_conv[-1].weight = 0
        # self.space_forward_conv[-1].bias = 1
        # self.view_fosrward_conv[-1].weight = 0
        # self.view_forward_conv[-1].bias = 1
        nn.init.zeros_(self.gbuffer_conditional_layer[-1].weight)
        nn.init.zeros_(self.gbuffer_conditional_layer[-1].bias)

    def kernel(self, direct_light_reprs, hard_shadow, shadow_feature):
        hard_shadow_image = hard_shadow.repeat(3, 1, 1, 1)
        compress_direct_repr = self.direct_compress_layer(direct_light_reprs)
        shadow_feature = shadow_feature.repeat(3, 1, 1, 1)

        # print("shadow feature :", shadow_feature.shape)
        # print("hard_shadow feature :", hard_shadow_image.shape)
        # print("compress_direct_repr feature :", compress_direct_repr.shape)
        # exit()
        shadow_result, weights,shadow_list = self.shadow_network.step(shadow_feature.permute(0, 3, 1, 2),
                                                      hard_shadow_image.permute(0, 3, 1, 2),
                                                      compress_direct_repr.permute(0, 3, 1, 2))
        return shadow_result, weights,shadow_list

    def get_lightformer_input(self, localLightData, localData):
        # print(localLightData['shadow'].keys())
        z_f = localLightData['shadow']['pixel_emitter_distance'] / 3
        z = localLightData['shadow']['occluder_emitter_distance'] / 3

        position_mask = localData["position_mask"]
        z_f[position_mask[..., 0]] = 3
        z[position_mask[..., 0]] = -3

        normal = localData["gbuffer"]["normal"]

        toLight = localData["toLight"][..., :3]
        toView = localData["gbuffer"]["view_dir"]
        c_c = torch.sum(localData["gbuffer"]['normal'] * localData["gbuffer"]['view_dir'], axis=-1)[
            ..., None]  # diffuse # TODO: Use this feature?
        depth = localData["gbuffer"]["depth"]
        localData["z_f_jian_z"] = z_f - z
        localData["z_f_chu_z"] = z_f / z
        localData["normal_dot_view"] = c_c
        localData["hard_shadow"] = torch.where(z_f - z > 1e-3, 0, 1)
        # pyexr.write("../testData/hard_shadow.exr",localData["hard_shadow"][0].cpu().numpy())
        # return torch.cat([z_f - z, z_f / z, depth,c_c,localData["hard_shadow"]], dim=-1)
        return torch.cat([z_f - z, z_f / z, depth[:1,...], c_c[:1,...]], dim=-1)

    def read_dir(self, resolution):
        dir_fn = "../datasets/standard_dir{}.pkl.zst".format(resolution)
        with open(dir_fn, 'rb') as gf:
            ##print(lightData_fn + self.data_type)
            dctx = zstd.ZstdDecompressor()
            dir_data = pickle.loads(dctx.decompress(gf.read()))
            ##print_dict(lightData)
        gf.close()
        return dir_data

    def fetch_tri_plane(self, triplane, xyz):
        xy = xyz[..., [0, 1]]
        xz = xyz[..., [0, 2]]
        yz = xyz[..., [1, 2]]
        xy_feature = torch.nn.functional.grid_sample(triplane[0:1], xy).permute(0, 2, 3, 1)
        xz_feature = torch.nn.functional.grid_sample(triplane[1:2], xz).permute(0, 2, 3, 1)
        yz_feature = torch.nn.functional.grid_sample(triplane[2:3], yz).permute(0, 2, 3, 1)

        return xy_feature + xz_feature + yz_feature

    def down_sampleing_plane(self, light_input, angular_size, space_size):
        B, W1, H1, W2, H2, C = light_input.shape
        angular_pool = nn.AvgPool2d(angular_size)
        space_pool = nn.AvgPool2d(space_size)
        light_input = space_pool(light_input.permute(0, 1, 2, 5, 3, 4).reshape(B * W1 * H1, C, W2, H2))
        W2, H2 = W2 // space_size, H2 // space_size
        light_input = light_input.reshape(B, W1, H1, C, W2, H2).permute(0, 4, 5, 3, 1, 2).reshape(B * W2 * H2, C, W1,
                                                                                                  H1)
        light_input = angular_pool(light_input)
        W1, H1 = W1 // angular_size, H1 // angular_size
        light_input = light_input.reshape(B, W2, H2, C, W1, H1).permute(0, 4, 5, 1, 2, 3)
        return light_input
    def preprocess_channel_cut(self,data):
        lightData = data["global"]
        lightData["radiance"] = lightData["radiance"].permute(-1, 1, 2, 3, 4, 0)
        lightData["position"] = lightData["position"].repeat(3,1,1,1,1,1)
        lightData["direction"] = lightData["direction"].repeat(3,1,1,1,1,1)
        localData = data["local"]
        for key in localData["gbuffer"]:
            localData["gbuffer"][key] = localData["gbuffer"][key].repeat(3,1,1,1)
        return data
    def forward(self, data, ligih_feature_list=None):
        SW1, SW2 = data["local"]["specular_shading"].shape[1:3]
        enable_timing_profile = True
        data["global"]["direction"] = self.dir_data
        data = self.preprocess_channel_cut(data)
        lightData = data["global"]
        localData = data["local"]
        light_input = torch.cat(
            [lightData["radiance"], lightData["position"],
             lightData["direction"]], dim=-1)

        light_down = self.down_sampleing_plane(light_input, 2, 2)
        # print_image_exr(light_down, "light_down")
        global_light_feature = self.global_light_encoder(light_down)

        principle_tri_plane = self.principle_triplane_encoder.forward_triplane(global_light_feature)
        # tri_plane_feature = self.triplane_encoder.forward_triplane(global_light_feature)

        specular_reflect_dir = torch.sum(data["local"]["gbuffer"]["view_dir"] * data["local"]["gbuffer"]["normal"],
                                         dim=-1).unsqueeze(-1) * data["local"]["gbuffer"]["normal"] * 2 - \
                               data["local"]["gbuffer"]["view_dir"]
        data["local"]["gbuffer"]["specular_ray"] = specular_reflect_dir
        data["local"]["gbuffer"]["half_vec"] = normalize(
            (normalize(-data["local"]["gbuffer"]["lposition"]) + data["local"]["gbuffer"]["view_dir"]) / 2)
        data["local"]["gbuffer"]["dot"] = torch.sum(
            data["local"]["gbuffer"]["half_vec"] * data["local"]["gbuffer"]["normal"], dim=-1, keepdim=True)
        W1, W2 = 32, 32
        #print("light_input ", light_input.shape)
        light_space_gbuffer = get_light_space_gbuffer(data)

        priciple_triplane_feature,voxel_coord = self.principle_triplane_encoder.fetch_spherical_tri_plane(principle_tri_plane, data["local"]["gbuffer"]["lposition"])

        l_lenth = torch.norm(data["local"]["gbuffer"]["lposition"], dim=-1, keepdim=True)

        data["local"]["sampled_principle_value"] = priciple_triplane_feature[..., :3]
        final_feature = priciple_triplane_feature
        shadow_input = self.get_lightformer_input(localData["lights"], localData)

        local_gbuffer_input = torch.cat(
            [light_space_gbuffer["normal"], light_space_gbuffer["half_vec"],
             light_space_gbuffer["specular_ray"],
             data["local"]["gbuffer"]["roughness"],
             l_lenth], dim=-1)
        #diffuse_gbuffers =torch.cat([shadow_input,local_gbuffer_input[:1,...]],dim=-1)
        diffuse_gbuffers =torch.cat([shadow_input],dim=-1)
        if torch.any(shadow_input.isnan()):
            print("shadow input nan")
            exit()
        shadow_input = torch.cat([shadow_input], dim=-1)
        #shadow_input = torch.cat([shadow_input, local_gbuffer_input[:1,...]], dim=-1)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        shadow_result, weights, shadow_list = eval(
            "self.{}(final_feature,localData[\"hard_shadow\"],shadow_input)".format("kernel"))
        result = {}
        result["shadow"] = shadow_result
        result["direct_shading"] = localData["direct_shading"] * result["shadow"]
        for i in range(5):
            data["local"]["single_shadow_{}".format(i)] = shadow_list[i]
        ender.record()
        torch.cuda.synchronize()

        # shadow_result = 1 - shadow_result
        #print("shadow inference time: ", starter.elapsed_time(ender))
        localData["shadow_weight0"] = weights[..., :3]
        localData["shadow_weight1"] = weights[..., 3:]

        result["shadow"] = torch.where(
            data["local"]["instance_mask"][..., :1],
            localData["shadow"],
            result["shadow"])

        if torch.any(torch.isnan(result["shadow"])):
            print("nan")
            exit()
        result["specular_shading"] = localData["specular_shading"]
        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)
        return data, result, loss_map, {}


class AccDiffuseDecoder(nn.Module):
    def __init__(self, configs, loss_config):
        super().__init__()
        #self.roughness_table = nn.Parameter(get_roughness_angular_table().unsqueeze(dim=-1), requires_grad=False)

        self.configs = configs
        self.relative = configs["relative"]
        self.angular_size = 32
        self.space_size = 64
        self.local_gbuffer = configs["local_gbuffer"]
        self.loss_config = loss_config

        self.angular_pool = [1, 2, 4, 8]
        # self.space_pool = [1,4,16,64]
        self.space_pool = [1, 2, 4, 8]
        self.channel_cut = True

        if self.channel_cut:
            self.principle_triplane_encoder = ChannelCutDiTransformerPlane(configs)
        else:
            self.principle_triplane_encoder = SimpleDiTransformerPlane(configs)
        # self.triplane_encoder = MultiScaleDiTransformerPlane(configs)

        self.light_input_dim = 9

        if self.channel_cut:
            self.light_input_dim = 7
        self.global_light_encoder = FourDTransformer(angular_size=self.angular_size // self.angular_pool[1],
                                                     space_size=self.space_size // self.space_pool[1],
                                                     angular_block_size=2,
                                                     space_block_size=8,
                                                     num_layers=configs["attention_layer_cnt"],
                                                     num_heads=16,
                                                     input_dim=self.light_input_dim,
                                                     hidden_dim=configs["plane"]["encoder_direct_dim"],
                                                     mlp_dim=configs["plane"]["encoder_direct_dim"] * 4,
                                                     output_dim=configs["plane"]["encoder_direct_dim"],
                                                     linear_output=False
                                                     )
        self.specular_feature = configs["plane"]["decoder_direct_dim"]
        # self.key_encoder = nn.Sequential(
        #     fc_layer(8, self.specular_feature),
        #     fc_layer(self.specular_feature, self.specular_feature),
        #     fc_layer(self.specular_feature, self.specular_feature)
        # )
        # self.attention_sample_4ds = CrossAttentionBlock(inner_dim=self.specular_feature,
        #                                                 cond_dim=self.specular_feature,
        #                                                 num_heads=16, eps=1e-6)

        self.loss_func = Loss(loss_config) if loss_config else None

        specular_feature = self.specular_feature

        # self.light_encoder = nn.Sequential(
        #     fc_layer(9, self.specular_feature * 1),
        #     fc_layer(self.specular_feature * 1, self.specular_feature * 1)
        # )
        # self.conv_encoder = LightFieldUNet(in_channels=7, base_channels=self.specular_feature)
        self.output_dim = 3
        self.conditional_layer = False
        if self.channel_cut:
            self.output_dim = 1
        if self.local_gbuffer:
            if self.conditional_layer:
                self.decoder = nn.Sequential(
                    fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.output_dim))
                self.gbuffer_conditional_layer = nn.Sequential(fc_layer(12,32),
                                                               fc_layer(32,32),
                                                               nn.Linear(32,self.specular_feature * 2))
                self.init()
            else:
                self.decoder = nn.Sequential(
                    fc_layer(self.specular_feature * 1 + 12, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.output_dim))

        else:
            self.decoder = nn.Sequential(
                fc_layer(self.specular_feature * 1 + 15, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.output_dim))
        self.dir_data = self.read_dir(str(self.angular_size) + "x" + str(self.space_size))
        self.dir_data = nn.Parameter(torch.Tensor(self.dir_data["direction"] / \
                                                  numpy.linalg.norm(self.dir_data["direction"], axis=-1)[
                                                      ..., numpy.newaxis]).unsqueeze(0), requires_grad=False)
        #self.shadow_network = KernelShadowNetwork(48)
        #self.direct_compress_layer = nn.Linear(64, 32)
    def init(self):
        # self.space_forward_conv[-1].weight = 0
        # self.space_forward_conv[-1].bias = 1
        # self.view_fosrward_conv[-1].weight = 0
        # self.view_forward_conv[-1].bias = 1
        nn.init.zeros_(self.gbuffer_conditional_layer[-1].weight)
        nn.init.zeros_(self.gbuffer_conditional_layer[-1].bias)

    def kernel(self, direct_light_reprs, hard_shadow, shadow_feature):
        hard_shadow_image = hard_shadow.repeat(3, 1, 1, 1)
        compress_direct_repr = self.direct_compress_layer(direct_light_reprs)
        shadow_feature = shadow_feature.repeat(3, 1, 1, 1)

        # print("shadow feature :", shadow_feature.shape)
        # print("hard_shadow feature :", hard_shadow_image.shape)
        # print("compress_direct_repr feature :", compress_direct_repr.shape)
        # exit()
        shadow_result, weights = self.shadow_network.step(shadow_feature.permute(0, 3, 1, 2),
                                                      hard_shadow_image.permute(0, 3, 1, 2),
                                                      compress_direct_repr.permute(0, 3, 1, 2))
        return shadow_result, weights

    def get_lightformer_input(self, localLightData, localData):
        # print(localLightData['shadow'].keys())
        z_f = localLightData['shadow']['pixel_emitter_distance'] / 3
        z = localLightData['shadow']['occluder_emitter_distance'] / 3

        position_mask = localData["position_mask"]
        z_f[position_mask[..., 0]] = 3
        z[position_mask[..., 0]] = -3

        normal = localData["gbuffer"]["normal"]

        toLight = localData["toLight"][..., :3]
        toView = localData["gbuffer"]["view_dir"]
        c_c = torch.sum(localData["gbuffer"]['normal'] * localData["gbuffer"]['view_dir'], axis=-1)[
            ..., None]  # diffuse # TODO: Use this feature?
        depth = localData["gbuffer"]["depth"]
        localData["z_f_jian_z"] = z_f - z
        localData["z_f_chu_z"] = z_f / z
        localData["normal_dot_view"] = c_c
        localData["hard_shadow"] = torch.where(z_f - z > 1e-3, 0, 1)
        # pyexr.write("../testData/hard_shadow.exr",localData["hard_shadow"][0].cpu().numpy())
        # return torch.cat([z_f - z, z_f / z, depth,c_c,localData["hard_shadow"]], dim=-1)
        return torch.cat([z_f - z, z_f / z, depth[:1,...], c_c[:1,...]], dim=-1)

    def read_dir(self, resolution):
        dir_fn = "../datasets/standard_dir{}.pkl.zst".format(resolution)
        with open(dir_fn, 'rb') as gf:
            ##print(lightData_fn + self.data_type)
            dctx = zstd.ZstdDecompressor()
            dir_data = pickle.loads(dctx.decompress(gf.read()))
            ##print_dict(lightData)
        gf.close()
        return dir_data

    def fetch_tri_plane(self, triplane, xyz):
        xy = xyz[..., [0, 1]]
        xz = xyz[..., [0, 2]]
        yz = xyz[..., [1, 2]]
        xy_feature = torch.nn.functional.grid_sample(triplane[0:1], xy).permute(0, 2, 3, 1)
        xz_feature = torch.nn.functional.grid_sample(triplane[1:2], xz).permute(0, 2, 3, 1)
        yz_feature = torch.nn.functional.grid_sample(triplane[2:3], yz).permute(0, 2, 3, 1)

        return xy_feature + xz_feature + yz_feature

    def down_sampleing_plane(self, light_input, angular_size, space_size):
        B, W1, H1, W2, H2, C = light_input.shape
        angular_pool = nn.AvgPool2d(angular_size)
        space_pool = nn.AvgPool2d(space_size)
        light_input = space_pool(light_input.permute(0, 1, 2, 5, 3, 4).reshape(B * W1 * H1, C, W2, H2))
        W2, H2 = W2 // space_size, H2 // space_size
        light_input = light_input.reshape(B, W1, H1, C, W2, H2).permute(0, 4, 5, 3, 1, 2).reshape(B * W2 * H2, C, W1,
                                                                                                  H1)
        light_input = angular_pool(light_input)
        W1, H1 = W1 // angular_size, H1 // angular_size
        light_input = light_input.reshape(B, W2, H2, C, W1, H1).permute(0, 4, 5, 1, 2, 3)
        return light_input
    def preprocess_channel_cut(self,data):
        lightData = data["global"]
        lightData["radiance"] = lightData["radiance"].permute(-1, 1, 2, 3, 4, 0)
        lightData["position"] = lightData["position"].repeat(3,1,1,1,1,1)
        lightData["direction"] = lightData["direction"].repeat(3,1,1,1,1,1)
        localData = data["local"]
        for key in localData["gbuffer"]:
            localData["gbuffer"][key] = localData["gbuffer"][key].repeat(3,1,1,1)
        return data
    def forward(self, data, ligih_feature_list=None):
        SW1, SW2 = data["local"]["specular_shading"].shape[1:3]
        enable_timing_profile = True
        data["global"]["direction"] = self.dir_data
        data = self.preprocess_channel_cut(data)
        lightData = data["global"]
        localData = data["local"]
        light_input = torch.cat(
            [lightData["radiance"], lightData["position"],
             lightData["direction"]], dim=-1)

        light_down = self.down_sampleing_plane(light_input, 2, 2)
        # print_image_exr(light_down, "light_down")
        global_light_feature = self.global_light_encoder(light_down)

        principle_tri_plane = self.principle_triplane_encoder.forward_triplane(global_light_feature)
        # tri_plane_feature = self.triplane_encoder.forward_triplane(global_light_feature)

        specular_reflect_dir = torch.sum(data["local"]["gbuffer"]["view_dir"] * data["local"]["gbuffer"]["normal"],
                                         dim=-1).unsqueeze(-1) * data["local"]["gbuffer"]["normal"] * 2 - \
                               data["local"]["gbuffer"]["view_dir"]
        data["local"]["gbuffer"]["specular_ray"] = specular_reflect_dir
        data["local"]["gbuffer"]["half_vec"] = normalize(
            (normalize(-data["local"]["gbuffer"]["lposition"]) + data["local"]["gbuffer"]["view_dir"]) / 2)
        data["local"]["gbuffer"]["dot"] = torch.sum(
            data["local"]["gbuffer"]["half_vec"] * data["local"]["gbuffer"]["normal"], dim=-1, keepdim=True)
        W1, W2 = 32, 32
        print("light_input ", light_input.shape)
        light_space_gbuffer = get_light_space_gbuffer(data)

        priciple_triplane_feature,voxel_coord = self.principle_triplane_encoder.fetch_spherical_tri_plane(principle_tri_plane, data["local"]["gbuffer"]["lposition"])

        l_lenth = torch.norm(data["local"]["gbuffer"]["lposition"], dim=-1, keepdim=True)

        data["local"]["sampled_principle_value"] = priciple_triplane_feature[..., :3]
        final_feature = priciple_triplane_feature

        local_gbuffer_input = torch.cat(
            [light_space_gbuffer["normal"], light_space_gbuffer["half_vec"],
             light_space_gbuffer["specular_ray"],
             data["local"]["gbuffer"]["roughness"],
                          data["local"]["gbuffer"]["dot"],
             l_lenth], dim=-1)

        diffuse_input = torch.cat([final_feature, local_gbuffer_input[:,...]], dim=-1)

        diffuse_shading = self.decoder(diffuse_input)
        if self.channel_cut:
            diffuse_shading = diffuse_shading.permute(-1, 1, 2, 0)
        localData = data["local"]
        result = {}
        result["log1p_diffuse_shading"] = diffuse_shading
        localData["pred_log1p_diffuse_shading"] = diffuse_shading

        result["log1p_diffuse_shading"] = torch.where(
            data["local"]["mask"],
            localData["log1p_diffuse_shading"],
            result["log1p_diffuse_shading"])

        if not self.relative:
            result["diffuse_shading"] = expp1_torch(result["log1p_diffuse_shading"]) * data["local"]["gbuffer"][
                "albedo"] * data["global"]["max_scale"]

            data["local"]["diffuse_shading"] = data["local"]["diffuse_shading"] * data["local"]["gbuffer"][
                "albedo"] * \
                                                data["global"]["max_scale"]
        else:
            # result["direct_shading"] = result["diffuse_shading"] + data["local"]["direct_shading"] - data["local"]["diffuse_shading"]
            result["diffuse_shading"] = expp1_torch(result["log1p_diffuse_shading"]) * data["local"]["gbuffer"][
                "albedo"][:1,...] * data["global"]["max_scale"] / data["local"]["relative"]
            data["local"]["diffuse_shading"] = data["local"]["diffuse_shading"] * data["local"]["gbuffer"][
                "albedo"][:1,...] * data["global"]["max_scale"] / data["local"]["relative"]

        if torch.any(torch.isnan(result["diffuse_shading"])):
            print("nan")
            exit()

        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)
        return data, result, loss_map, {}




class FinalDecoder(nn.Module):
    def __init__(self, configs, loss_config):
        super().__init__()
        self.roughness_table = nn.Parameter(get_roughness_angular_table().unsqueeze(dim=-1), requires_grad=False)

        self.configs = configs
        self.relative = configs["relative"]
        self.angular_size = 32
        self.space_size = 64
        self.local_gbuffer = configs["local_gbuffer"]
        self.loss_config = loss_config

        self.angular_pool = [1, 2, 4, 8]
        # self.space_pool = [1,4,16,64]
        self.space_pool = [1, 2, 4, 8]
        self.channel_cut = True

        if self.channel_cut:
            self.principle_triplane_encoder = ChannelCutDiTransformerPlane(configs)
        else:
            self.principle_triplane_encoder = SimpleDiTransformerPlane(configs)
        # self.triplane_encoder = MultiScaleDiTransformerPlane(configs)
        self.specular_feature = configs["specular_dim"]
        self.light_input_dim = 9

        if self.channel_cut:
            self.light_input_dim = 7
        self.global_light_encoder = FourDTransformer(
                                                        angular_size=self.angular_size // self.angular_pool[1],
                                                        space_size=self.space_size // self.space_pool[1],
                                                        angular_block_size=2,
                                                        space_block_size=8,
                                                        num_layers=configs["attention_layer_cnt"],
                                                        num_heads=16,
                                                        input_dim=self.light_input_dim,
                                                        hidden_dim=self.specular_feature * 4,
                                                        mlp_dim=self.specular_feature * 1,
                                                        output_dim=self.specular_feature * 2,
                                                        linear_output=False,
                                                     )
        self.specular_feature = configs["plane"]["decoder_direct_dim"]
        # self.key_encoder = nn.Sequential(
        #     fc_layer(8, self.specular_feature),
        #     fc_layer(self.specular_feature, self.specular_feature),
        #     fc_layer(self.specular_feature, self.specular_feature)
        # )
        # self.attention_sample_4ds = CrossAttentionBlock(inner_dim=self.specular_feature,
        #                                                 cond_dim=self.specular_feature,
        #                                                 num_heads=16, eps=1e-6)

        self.loss_func = Loss(loss_config) if loss_config else None

        specular_feature = self.specular_feature

        # self.light_encoder = nn.Sequential(
        #     fc_layer(9, self.specular_feature * 1),
        #     fc_layer(self.specular_feature * 1, self.specular_feature * 1)
        # )
        # self.conv_encoder = LightFieldUNet(in_channels=7, base_channels=self.specular_feature)
        self.output_dim = 3
        self.conditional_layer = False
        if self.channel_cut:
            self.output_dim = 1
        if self.local_gbuffer:
            if self.conditional_layer:
                self.decoder = nn.Sequential(
                    fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.output_dim))
                self.gbuffer_conditional_layer = nn.Sequential(fc_layer(12,32),
                                                               fc_layer(32,32),
                                                               nn.Linear(32,self.specular_feature * 2))
                self.init()
            else:
                self.decoder = nn.Sequential(
                    fc_layer(self.specular_feature * 1 + 12, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.output_dim))

        else:
            self.decoder = nn.Sequential(
                fc_layer(self.specular_feature * 1 + 15, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.output_dim))
        self.dir_data = self.read_dir(str(self.angular_size) + "x" + str(self.space_size))
        self.dir_data = nn.Parameter(torch.Tensor(self.dir_data["direction"] / \
                                                  numpy.linalg.norm(self.dir_data["direction"], axis=-1)[
                                                      ..., numpy.newaxis]).unsqueeze(0), requires_grad=False)
        self.shadow_network = SmallKernelShadowNetwork(21)
        self.direct_compress_layer = nn.Linear(self.specular_feature, 16)
        self.diffuse_decoder = nn.Sequential(
                    fc_layer(self.specular_feature * 1 + 12, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                    fc_layer(self.specular_feature * 1, self.output_dim))
        self.transport = configs["plane"]["transport"]


    def init(self):
        # self.space_forward_conv[-1].weight = 0
        # self.space_forward_conv[-1].bias = 1
        # self.view_fosrward_conv[-1].weight = 0
        # self.view_forward_conv[-1].bias = 1
        nn.init.zeros_(self.gbuffer_conditional_layer[-1].weight)
        nn.init.zeros_(self.gbuffer_conditional_layer[-1].bias)

    def kernel(self, direct_light_reprs, hard_shadow, shadow_feature):
        hard_shadow_image = hard_shadow.repeat(3, 1, 1, 1)
        compress_direct_repr = self.direct_compress_layer(direct_light_reprs)
        shadow_feature = shadow_feature.repeat(3, 1, 1, 1)

        # print("shadow feature :", shadow_feature.shape)
        # print("hard_shadow feature :", hard_shadow_image.shape)
        # print("compress_direct_repr feature :", compress_direct_repr.shape)
        # exit()
        shadow_result, weights,shadow_list = self.shadow_network.step(shadow_feature.permute(0, 3, 1, 2),
                                                      hard_shadow_image.permute(0, 3, 1, 2),
                                                      compress_direct_repr.permute(0, 3, 1, 2))
        return shadow_result, weights,shadow_list

    def get_lightformer_input(self, localLightData, localData):
        # print(localLightData['shadow'].keys())
        z_f = localLightData['shadow']['pixel_emitter_distance'] / 3
        z = localLightData['shadow']['occluder_emitter_distance'] / 3

        position_mask = localData["position_mask"]
        z_f[position_mask[..., 0]] = 3
        z[position_mask[..., 0]] = -3

        normal = localData["gbuffer"]["normal"]

        toLight = localData["toLight"][..., :3]
        toView = localData["gbuffer"]["view_dir"]
        c_c = torch.sum(localData["gbuffer"]['normal'] * localData["gbuffer"]['view_dir'], axis=-1)[
            ..., None]  # diffuse # TODO: Use this feature?
        depth = localData["gbuffer"]["depth"]
        localData["z_f_jian_z"] = z_f - z
        localData["z_f_chu_z"] = z_f / z
        localData["normal_dot_view"] = c_c
        localData["hard_shadow"] = torch.where(z_f - z > 1e-3, 0, 1)
        # pyexr.write("../testData/hard_shadow.exr",localData["hard_shadow"][0].cpu().numpy())
        # return torch.cat([z_f - z, z_f / z, depth,c_c,localData["hard_shadow"]], dim=-1)
        return torch.cat([z_f - z, z_f / z, depth[:1,...], c_c[:1,...]], dim=-1)


    def read_dir(self, resolution):
        dir_fn = "../datasets/standard_dir{}.pkl.zst".format(resolution)
        with open(dir_fn, 'rb') as gf:
            ##print(lightData_fn + self.data_type)
            dctx = zstd.ZstdDecompressor()
            dir_data = pickle.loads(dctx.decompress(gf.read()))
            ##print_dict(lightData)
        gf.close()
        return dir_data

    def fetch_tri_plane(self, triplane, xyz):
        xy = xyz[..., [0, 1]]
        xz = xyz[..., [0, 2]]
        yz = xyz[..., [1, 2]]
        xy_feature = torch.nn.functional.grid_sample(triplane[0:1], xy).permute(0, 2, 3, 1)
        xz_feature = torch.nn.functional.grid_sample(triplane[1:2], xz).permute(0, 2, 3, 1)
        yz_feature = torch.nn.functional.grid_sample(triplane[2:3], yz).permute(0, 2, 3, 1)

        return xy_feature + xz_feature + yz_feature

    def down_sampleing_plane(self, light_input, angular_size, space_size):
        B, W1, H1, W2, H2, C = light_input.shape
        angular_pool = nn.AvgPool2d(angular_size)
        space_pool = nn.AvgPool2d(space_size)
        light_input = space_pool(light_input.permute(0, 1, 2, 5, 3, 4).reshape(B * W1 * H1, C, W2, H2))
        W2, H2 = W2 // space_size, H2 // space_size
        light_input = light_input.reshape(B, W1, H1, C, W2, H2).permute(0, 4, 5, 3, 1, 2).reshape(B * W2 * H2, C, W1,
                                                                                                  H1)
        light_input = angular_pool(light_input)
        W1, H1 = W1 // angular_size, H1 // angular_size
        light_input = light_input.reshape(B, W2, H2, C, W1, H1).permute(0, 4, 5, 1, 2, 3)
        return light_input
    def preprocess_channel_cut(self,data):
        lightData = data["global"]
        lightData["radiance"] = lightData["radiance"].permute(-1, 1, 2, 3, 4, 0)
        lightData["position"] = lightData["position"].repeat(3,1,1,1,1,1)
        lightData["direction"] = lightData["direction"].repeat(3,1,1,1,1,1)
        localData = data["local"]
        for key in localData["gbuffer"]:
            localData["gbuffer"][key] = localData["gbuffer"][key].repeat(3,1,1,1)
        return data
    def forward(self, data, ligih_feature_list=None):
        SW1, SW2 = data["local"]["specular_shading"].shape[1:3]
        enable_timing_profile = True
        data["global"]["direction"] = self.dir_data
        data = self.preprocess_channel_cut(data)
        lightData = data["global"]
        light_input = torch.cat(
            [lightData["radiance"], lightData["position"],
             lightData["direction"]], dim=-1)

        light_down = self.down_sampleing_plane(light_input, 2, 2)
        # print_image_exr(light_down, "light_down")
        global_light_feature = self.global_light_encoder(light_down)

        principle_tri_plane = self.principle_triplane_encoder.forward_triplane(global_light_feature)
        # tri_plane_feature = self.triplane_encoder.forward_triplane(global_light_feature)

        specular_reflect_dir = torch.sum(data["local"]["gbuffer"]["view_dir"] * data["local"]["gbuffer"]["normal"],
                                         dim=-1).unsqueeze(-1) * data["local"]["gbuffer"]["normal"] * 2 - \
                               data["local"]["gbuffer"]["view_dir"]
        data["local"]["gbuffer"]["specular_ray"] = specular_reflect_dir
        data["local"]["gbuffer"]["half_vec"] = normalize(
            (normalize(-data["local"]["gbuffer"]["lposition"]) + data["local"]["gbuffer"]["view_dir"]) / 2)
        data["local"]["gbuffer"]["dot"] = torch.sum(
            data["local"]["gbuffer"]["half_vec"] * data["local"]["gbuffer"]["normal"], dim=-1, keepdim=True)
        W1, W2 = 32, 32
        #print("light_input ", light_input.shape)
        # light_feature = self.conv_encoder(light_input)
        # light_feature = torch.nn.functional.leaky_relu(light_feature, 0.1)

        # lod_local_data_new = get_lod_tracing_arbitrary(data["local"]["gbuffer"]["lposition"], specular_reflect_dir,
        #                                                data["local"]["gbuffer"]["normal"],
        #                                                1 - data["local"]["gbuffer"]["roughness"],
        #                                                self.roughness_table, W1, W2)
        light_space_gbuffer = get_light_space_gbuffer(data)
        # sample_index = torch.cat([lod_local_data_new["angular_uv_left"], lod_local_data_new["space_uv_left"],
        #                           lod_local_data_new["angular_uv_right"], lod_local_data_new["space_uv_right"]], dim=-1)

        # sample_index[..., :4] = torch.where(torch.isnan(sample_index[..., :4]), 0, sample_index[..., :4])
        # sample_index[..., 4:] = torch.where(torch.isnan(sample_index[..., 4:]), 1, sample_index[..., 4:])

        # sample_index = sample_index.long()
        # sampled_value = batch_get_4d_region_sum(compute_4d_prefix_sum(light_feature[0]),
        #                                         sample_index[0].reshape(-1, 8)).reshape(1, SW1, SW2,
        #                                                                                 self.specular_feature)
        # lod_local_data_new["cone_scale2"] = torch.where(torch.isnan(lod_local_data_new["cone_scale2"]), 1,
        #                                                 lod_local_data_new["cone_scale2"])
        # area = sample_index[..., 4:] - sample_index[..., :4]
        # area = area[..., :1] * area[..., 1:2] * area[..., 2:3] * area[..., 3:4]
        # area = torch.clamp(area, min=1)
        # area_after_scale = area * lod_local_data_new["cone_scale"]
        # area_after_scale2 = area * lod_local_data_new["cone_scale2"]
        # sampled_value = sampled_value / area_after_scale2

        # triplane_index = torch.cat([lod_local_data_new["x"], lod_local_data_new["y"], lod_local_data_new["z"]], dim=-1)
        priciple_triplane_index = data["local"]["gbuffer"]["lposition"] / 6
        # data["local"]["triplane_index"] = triplane_index
        # triplane_feature = self.fetch_tri_plane(tri_plane_feature, triplane_index)
        # priciple_triplane_feature,coord = self.principle_triplane_encoder.fetch_spherical_tri_plane(principle_tri_plane,data["local"]["gbuffer"]["lposition"])
        # print_image_exr((coord+1)/2 * 64,"coord")
        if self.transport == "triplane_s":
            priciple_triplane_feature,voxel_coord = self.principle_triplane_encoder.fetch_spherical_tri_plane(principle_tri_plane, data["local"]["gbuffer"]["lposition"])
        elif self.transport == "triplane":
            priciple_triplane_feature,voxel_coord = self.principle_triplane_encoder.fetch_tri_plane(principle_tri_plane, data["local"]["gbuffer"]["lposition"])
        # priciple_triplane_feature = torch.nn.functional.leaky_relu(priciple_triplane_feature)
        # print_image_exr(triplane_index,"triplane_index")
        # print_image_exr(priciple_triplane_index,"priciple_triplane_index")
        data["local"]["feature_map"] = priciple_triplane_feature[...,:3]
        data["local"]["voxel_index"] = (voxel_coord + 1)/2 *32 // 1 /32
        # if torch.any(torch.isnan(sampled_value)):
        #     print("nan")
        # ##print_image_exr(lod_local_data_new["cone_scale2"],"cone_scale2")
        # ##print_image_exr(area_after_scale2[...,:3],"area_after_scale2")

        l_lenth = torch.norm(data["local"]["gbuffer"]["lposition"], dim=-1, keepdim=True)

        ##print_image_exr(mask,"mask")
        # data["local"]["sampled_value"] = sampled_value[..., :3]
        # data["local"]["sampled_triplane_value"] = triplane_feature[..., :3]
        data["local"]["sampled_principle_value"] = priciple_triplane_feature[..., :3]
        final_feature = priciple_triplane_feature
        # final_feature = sampled_value + triplane_feature
        # final_feature = priciple_triplane_feature
        # final_feature = triplane_feature
        # final_feature = sampled_value + triplane_feature
        # final_feature = triplane_feature + priciple_triplane_feature
        # final_feature = triplane_feature + priciple_triplane_feature
        if self.local_gbuffer:
            specular_input = torch.cat(
                [final_feature, light_space_gbuffer["normal"], light_space_gbuffer["half_vec"],
                 light_space_gbuffer["specular_ray"],
                 data["local"]["gbuffer"]["roughness"],
                 data["local"]["gbuffer"]["dot"],
                 l_lenth], dim=-1)
            local_gbuffer_input = torch.cat(
                [light_space_gbuffer["normal"], light_space_gbuffer["half_vec"],
                 light_space_gbuffer["specular_ray"],
                 data["local"]["gbuffer"]["roughness"],
                 data["local"]["gbuffer"]["dot"],
                 l_lenth], dim=-1)

            if self.conditional_layer:
                scale,shift = self.gbuffer_conditional_layer(local_gbuffer_input).chunk(2, dim=-1)
                print("scale minmax")
                print(scale.min())
                print(scale.max())
                print("shift minmax")
                print(shift.min())
                print(shift.max())
                specular_input =final_feature + final_feature * scale + shift
        else:
            specular_input = torch.cat(
                [final_feature, data["local"]["toLight"][..., :3], data["local"]["gbuffer"]["lposition"],
                 data["local"]["gbuffer"]["normal"],
                 data["local"]["gbuffer"]["view_dir"],
                 data["local"]["gbuffer"]["roughness"],
                 data["local"]["gbuffer"]["dot"],
                 l_lenth], dim=-1
            )
        specular_shading = self.decoder(specular_input)
        diffuse_shading = self.diffuse_decoder(specular_input)
        if self.channel_cut:
            specular_shading = specular_shading.permute(-1,1,2,0)
            diffuse_shading = diffuse_shading.permute(-1,1,2,0)

        result = {}
        # print("pixel decoder now")
        localData = data["local"]
        shadow_input = self.get_lightformer_input(localData["lights"], localData)
        shadow_input = torch.cat([shadow_input], dim=-1)
        #shadow_input = torch.cat([shadow_input, local_gbuffer_input[:1,...]], dim=-1)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        shadow_result, weights, shadow_list = eval(
            "self.{}(final_feature,localData[\"hard_shadow\"],shadow_input)".format("kernel"))
        
        result["shadow"] = shadow_result

        localData = data["local"]
        
        # mask = (data["local"]["gbuffer"]["roughness"] < 0.75).repeat(1,1,1,3)
        result["log1p_diffuse_shading"] = diffuse_shading

        result["log1p_diffuse_shading"] = torch.where(
            data["local"]["instance_mask"][..., :1] | data["local"]["albedo_mask"],
            localData["log1p_diffuse_shading"],
            result["log1p_diffuse_shading"])

        result["log1p_specular_shading"] = specular_shading

        result["log1p_specular_shading"] = torch.where(
            data["local"]["instance_mask"][..., :1] | data["local"]["specular_mask"],
            localData["log1p_specular_shading"],
            result["log1p_specular_shading"])

        localData["mask"] = data["local"]["instance_mask"][..., :1] | data["local"]["specular_mask"]
        # print("result max ",result["log1p_specular_shading"].max())
        # print("ref max ",localData["log1p_specular_shading"].max())
        if torch.any(torch.isnan(result["log1p_specular_shading"])):
            print("nan")
            exit()
        if not self.relative:
            result["specular_shading"] = expp1_torch(result["log1p_specular_shading"]) * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"]

            data["local"]["specular_shading"] = data["local"]["specular_shading"] * data["local"]["gbuffer"][
                "specular"] * \
                                                data["global"]["max_scale"]
            result["diffuse_shading"] = expp1_torch(result["log1p_diffuse_shading"]) * data["local"]["gbuffer"][
                "albedo"] * data["global"]["max_scale"]

            data["local"]["diffuse_shading"] = data["local"]["diffuse_shading"] * data["local"]["gbuffer"][
                "albedo"] * \
                                                data["global"]["max_scale"]
        else:
            # result["direct_shading"] = result["specular_shading"] + data["local"]["direct_shading"] - data["local"]["specular_shading"]
            result["specular_shading"] = expp1_torch(result["log1p_specular_shading"]) * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"] / data["local"]["relative"]
            data["local"]["specular_shading"] = data["local"]["specular_shading"] * data["local"]["gbuffer"][
                "specular"] * data["global"]["max_scale"] / data["local"]["relative"]

            result["diffuse_shading"] = expp1_torch(result["log1p_diffuse_shading"]) * data["local"]["gbuffer"][
                "albedo"] * data["global"]["max_scale"] / data["local"]["relative"]
            data["local"]["diffuse_shading"] = data["local"]["diffuse_shading"] * data["local"]["gbuffer"][
                "albedo"] * data["global"]["max_scale"] / data["local"]["relative"]
        # result["specular_shading"] = result["specular_shading"].repeat(1, 1, 1, 3)
        # data["local"]["specular_shading"] = data["local"]["specular_shading"].repeat(1, 1, 1, 3)
        # result["shadow"] = localData["shadow"]
        # result["log1p_diffuse_shading"] = localData["log1p_diffuse_shading"]
        # result["diffuse_shading"] = expp1_torch(result["log1p_diffuse_shading"])
        # #print("result: ",result["log1p_specular_shading"].min())
        # #print("gt: ",data["local"]["log1p_specular_shading"].min())
        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)
        return data, result, loss_map, {}


def relative_pos_to_cubemap_uv(rel_pos):
    """
    rel_pos: (N,3) tensor, 相对光源位置，即 dir = pos - light_pos
    return:
        face: (N,) int, 0~5
        uv: (N,2) float32 in [0,1]
    """
    x, y, z = rel_pos.unbind(-1)   # (N,)
    abs_x, abs_y, abs_z = rel_pos.abs().unbind(-1)
    max_axis = torch.stack([abs_x, abs_y, abs_z], dim=-1).argmax(dim=-1)  # (N,)
    sx, sy, sz = torch.sign(rel_pos).unbind(-1)

    # 预分配结果
    face = torch.zeros_like(max_axis)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)

    # === +X / -X ===
    mask_x = (max_axis == 0)
    pos_x = mask_x & (sx > 0)
    neg_x = mask_x & (sx < 0)

    face[pos_x] = 0  # +X
    u[pos_x] = -z[pos_x] / (abs_x[pos_x] + 1e-8)
    v[pos_x] =  y[pos_x] / (abs_x[pos_x] + 1e-8)

    face[neg_x] = 1  # -X
    u[neg_x] =  z[neg_x] / (abs_x[neg_x] + 1e-8)
    v[neg_x] =  y[neg_x] / (abs_x[neg_x] + 1e-8)

    # === +Y / -Y ===
    mask_y = (max_axis == 1)
    pos_y = mask_y & (sy > 0)
    neg_y = mask_y & (sy < 0)

    face[pos_y] = 2  # +Y
    u[pos_y] =  x[pos_y] / (abs_y[pos_y] + 1e-8)
    v[pos_y] = -z[pos_y] / (abs_y[pos_y] + 1e-8)

    face[neg_y] = 3  # -Y
    u[neg_y] =  x[neg_y] / (abs_y[neg_y] + 1e-8)
    v[neg_y] =  z[neg_y] / (abs_y[neg_y] + 1e-8)

    # === +Z / -Z ===
    mask_z = (max_axis == 2)
    pos_z = mask_z & (sz > 0)
    neg_z = mask_z & (sz < 0)

    face[pos_z] = 4  # +Z
    u[pos_z] =  x[pos_z] / (abs_z[pos_z] + 1e-8)
    v[pos_z] =  y[pos_z] / (abs_z[pos_z] + 1e-8)

    face[neg_z] = 5  # -Z
    u[neg_z] = -x[neg_z] / (abs_z[neg_z] + 1e-8)
    v[neg_z] =  y[neg_z] / (abs_z[neg_z] + 1e-8)

    # [-1,1] → [0,1]
    uv = torch.stack([u, v], dim=-1)
    uv = (uv + 1.0) * 0.5
    return face, uv

def cubemap_sample(textures, face, uv):
    """
    从 cubemap 的六个面中采样 texel。

    参数:
        textures: (6, C, H, W) tensor，六张贴图
        face: (N,) long tensor，每个点属于哪一面 (0~5)
        uv: (N,2) tensor，归一化坐标 [0,1]

    返回:
        samples: (N, C) tensor
    """
    device = textures.device
    N = face.shape[0]
    C, H, W = textures.shape[1:]
    
    # 将 uv 从 [0,1] → [-1,1] 适配 grid_sample
    grid = uv * 2.0 - 1.0
    grid = grid.view(1, N, 1, 2)  # (1, N, 1, 2)

    # 逐面采样 (一次性并行计算)
    samples = torch.zeros((N, C), device=device)
    for f in range(6):
        mask = (face == f)
        if not mask.any():
            continue
        face_tex = textures[f].unsqueeze(0)  # (1,C,H,W)
        uv_f = uv[mask].unsqueeze(0).unsqueeze(2)  # (1,Nf,1,2)
        uv_f = uv_f * 2 - 1  # to [-1,1]
        # F.grid_sample expects (N,C,H,W), grid=(N,H,W,2)
        sampled = F.grid_sample(face_tex, uv_f.permute(0,2,1,3), align_corners=True,padding_mode="border")
        samples[mask] = sampled.squeeze(2).squeeze(0).T  # (C,Nf) -> (Nf,C)
    return samples

def normalize_depth_with_mask(depth, mask, eps=1e-4):
    """
    depth: (B, W, H, 1)
    mask:  (B, W, H, 1), 1 表示 mask 区域，0 表示有效区域
    return: (B, W, H, 1)
    """
    B = depth.shape[0]
    depth_flat = depth.reshape(B, -1)
    mask_flat = mask.reshape(B, -1)

    # 有效区域
    valid = (mask_flat == 0)

    # 为避免无效区域干扰，mask掉它们
    depth_valid = torch.where(valid, depth_flat, torch.full_like(depth_flat, float('inf')))
    min_vals = depth_valid.min(dim=1).values

    depth_valid = torch.where(valid, depth_flat, torch.full_like(depth_flat, float('-inf')))
    max_vals = depth_valid.max(dim=1).values

    # 防止max==min
    range_vals = (max_vals - min_vals).clamp_min(eps)

    # 广播
    min_vals = min_vals[:, None]
    range_vals = range_vals[:, None]

    # 归一化
    norm_depth = (depth_flat - min_vals) / range_vals

    # mask 区域赋 -1
    norm_depth = torch.where(valid, norm_depth, torch.full_like(norm_depth, -1.0))

    return norm_depth.reshape_as(depth)

class DownsampleBy4_To6x1(nn.Module):
    """
    输入:  x ∈ [B, C, W, H]，例如 [B, 64, 64, 384]
    过程:  3 层 4x4 卷积, stride=4, padding=0
          空间: (H,W): 384×64 -> 96×16 -> 24×4 -> 6×1
    输出:  全连接后标量 [B, 1]
    """
    def __init__(self, output_channels = 64,in_channels=64, channels=(128, 256, 512), act=nn.ReLU(inplace=True)):
        super().__init__()
        c1, c2, c3 = channels
        self.input_layer = fc_layer(in_channels+7,64)
        self.conv = nn.Sequential(
            nn.Conv2d(64, c1, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(inplace=True) if act is None else act,
            nn.Conv2d(c1, c2, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(inplace=True) if act is None else act,
            nn.Conv2d(c2, c3, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(inplace=True) if act is None else act,
        )
        # 经过三层后空间 = [B, c3, 6, 1]，拼接 6 个位置 -> 线性到标量
        self.fc = nn.Linear(c3 * 6, output_channels)

    def forward(self, x):
        # 你的输入是 [B, C, W, H] -> 转为 PyTorch 约定 [B, C, H, W]

        x = self.input_layer(x)
        x = x.permute(0, 3, 1, 2).contiguous()   # [B, C, 384, 64]
        y = self.conv(x)                         # [B, c3, 6, 1]
        y = y.flatten(start_dim=1)               # [B, c3*6]
        y = self.fc(y)                           # [B, 1]
        return y

class DownsampleBy4_To6x1_Strenth(nn.Module):
    """
    输入:  x ∈ [B, C, W, H]，例如 [B, 64, 64, 384]
    过程:  3 层 4x4 卷积, stride=4, padding=0
          空间: (H,W): 384×64 -> 96×16 -> 24×4 -> 6×1
    输出:  全连接后标量 [B, output_channels]
    """

    def __init__(self, output_channels=64, in_channels=64,
                 channels=(128, 256, 512)):
        super().__init__()
        c1, c2, c3 = channels

        # --------------------------
        # 1. 可学习 Position Embedding
        # --------------------------
        # shape = [384, 64, 64]
        self.pos_embed = nn.Parameter(
            torch.randn(384, 64, 64) * 0.01
        )




        # --------------------------
        # 2. 输入层
        # --------------------------
        self.input_layer = fc_layer(in_channels + 7, 64)

        # --------------------------
        # 3. 卷积部分
        # --------------------------
        act = nn.LeakyReLU(0.3, inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(64, c1, kernel_size=4, stride=4, padding=0, bias=True),
            act,
            nn.Conv2d(c1, c2, kernel_size=4, stride=4, padding=0, bias=True),
            act,
            nn.Conv2d(c2, c3, kernel_size=4, stride=4, padding=0, bias=True),
            act,
        )

        # --------------------------
        # 4. 输出全连接
        # --------------------------
        self.fc = nn.Linear(c3 * 6, output_channels)

    def forward(self, x):
        """
        x: [B, C, 384, 64]
        """
        B, C, H, W = x.shape   # H=384, W=64

        # --------------------------
        # 添加可学习 Position Embedding
        # --------------------------
        # pos_embed: [384,64,64]
        # → expand → [B,384,64,64]
        pos = self.pos_embed.unsqueeze(0).expand(B, -1, -1, -1)


        # 将位置编码加入输入
        x = self.input_layer(x)           # still [B, 64, 384, 64]

        x = x + pos

        # --------------------------
        # 原始流程
        # --------------------------
        
        x = x.permute(0, 3, 1, 2)         # [B,64,384,64] → [B,64,64,384]

        y = self.conv(x)                  # [B,c3,6,1]
        y = y.flatten(start_dim=1)        # [B,c3*6]
        y = self.fc(y)                    # [B, output_channels]
        return y


import time
def start_timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()                     # ⭐ 必须记录 start
        return start, end
    else:
        return time.perf_counter(), None

def end_timer(start, end=None):
    if isinstance(start, torch.cuda.Event):
        end.record()                       # ⭐ 必须记录 end
        torch.cuda.synchronize()
        return start.elapsed_time(end)     # 返回毫秒
    else:
        return (time.perf_counter() - start) * 1000.0


class IndirectFarwardProxy(nn.Module):
    def __init__(self,indirect_feature,diffuse_decoder):
        super().__init__()
        self.cnn_encoder = True
        self.indirect_feature = indirect_feature
        self.diffuse_decoder = diffuse_decoder
        if self.cnn_encoder:
            #self.indirect_encoder = DownsampleBy4_To6x1(self.indirect_feature,self.indirect_compress)
            self.indirect_encoder = DownsampleBy4_To6x1_Strenth(self.indirect_feature,0)
        else:
            self.indirect_encoder = nn.Sequential(
                fc_layer(self.indirect_feature * 1 + 7, self.indirect_feature * 1),
                fc_layer(self.indirect_feature * 1, self.indirect_feature * 1),
                fc_layer(self.indirect_feature * 1, self.indirect_feature * 1),
                fc_layer(self.indirect_feature * 1, self.indirect_feature * 1))
        self.indirect_decoder = nn.Sequential(
                fc_layer(self.indirect_feature * 1 + 13, self.indirect_feature * 1),
                fc_layer(self.indirect_feature * 1, self.indirect_feature * 1),
                fc_layer(self.indirect_feature * 1, self.indirect_feature * 1),
                fc_layer(self.indirect_feature * 1, 2))

    def forward(self,rsm_input,gbuffer_input):
        diffuse_decoder_input = rsm_input[...,:64 + 12]
        light_albedo,light_position,light_normal = rsm_input[...,64 + 12: 64 + 12 +1], rsm_input[...,64 + 12 +1: 64 + 12 +4], rsm_input[...,64 + 12+4: 64 + 12 +7]

        diffuse_direct_result = self.diffuse_decoder(diffuse_decoder_input) * light_albedo
        B,_,_,C = diffuse_direct_result.shape
        ma = diffuse_direct_result.reshape(B,-1,C).max(dim=1).values.unsqueeze(1).unsqueeze(1)
        normalize_diffuse_direct_result = diffuse_direct_result / ma
        indirect_encoder_input = torch.cat([normalize_diffuse_direct_result,light_position,light_normal],dim=-1)
        indirect_feature = self.indirect_encoder(indirect_encoder_input).unsqueeze(1).unsqueeze(1).repeat(1,512,512,1)
        indirect_decoder_input = torch.cat([indirect_feature,gbuffer_input],dim=-1)
        print(indirect_decoder_input.shape)
        print(ma.shape)
        indirect_result = self.indirect_decoder(indirect_decoder_input) * ma
        return indirect_result
    def train_step(self,rsm_input,gbuffer_input,gt,mask):
        diffuse_decoder_input = rsm_input[...,:64 + 12]
        light_albedo,light_position,light_normal = rsm_input[...,64 + 12: 64 + 12 +1], rsm_input[...,64 + 12 +1: 64 + 12 +4], rsm_input[...,64 + 12+4: 64 + 12 +7]

        diffuse_direct_result = self.diffuse_decoder(diffuse_decoder_input) * light_albedo
        B,_,_,C = diffuse_direct_result.shape
        ma = diffuse_direct_result.reshape(B,-1,C).max(dim=1).values.unsqueeze(1).unsqueeze(1)
        normalize_diffuse_direct_result = diffuse_direct_result / ma
        indirect_encoder_input = torch.cat([normalize_diffuse_direct_result,light_position,light_normal],dim=-1)
        indirect_feature = self.indirect_encoder(indirect_encoder_input).unsqueeze(1).unsqueeze(1).repeat(1,512,512,1)
        indirect_decoder_input = torch.cat([indirect_feature,gbuffer_input],dim=-1)
        indirect_result = self.indirect_decoder(indirect_decoder_input) * ma
        indirect_result = torch.where(mask,gt,indirect_result)
        return indirect_result,normalize_diffuse_direct_result
    def train_step(self,rsm_input,gbuffer_input,gt,mask):
        diffuse_decoder_input = rsm_input[...,:64 + 12]
        light_albedo,light_position,light_normal = rsm_input[...,64 + 12: 64 + 12 +1], rsm_input[...,64 + 12 +1: 64 + 12 +4], rsm_input[...,64 + 12+4: 64 + 12 +7]

        diffuse_direct_result = self.diffuse_decoder(diffuse_decoder_input) * light_albedo
        B,_,_,C = diffuse_direct_result.shape
        ma = diffuse_direct_result.reshape(B,-1,C).max(dim=1).values.unsqueeze(1).unsqueeze(1)
        normalize_diffuse_direct_result = diffuse_direct_result / ma
        indirect_encoder_input = torch.cat([normalize_diffuse_direct_result,light_position,light_normal],dim=-1)
        indirect_feature = self.indirect_encoder(indirect_encoder_input).unsqueeze(1).unsqueeze(1).repeat(1,512,512,1)
        indirect_decoder_input = torch.cat([indirect_feature,gbuffer_input],dim=-1)
        indirect_result = self.indirect_decoder(indirect_decoder_input) * ma
        indirect_result = torch.where(mask,gt,indirect_result)
        return indirect_result,normalize_diffuse_direct_result
    
    def train_step_nodiffuse(self,rsm_input,gbuffer_input,gt,mask):
        diffuse_decoder_input = rsm_input[...,:64 + 12]
        light_albedo,light_position,light_normal = rsm_input[...,64 + 12: 64 + 12 +1], rsm_input[...,64 + 12 +1: 64 + 12 +4], rsm_input[...,64 + 12+4: 64 + 12 +7]

        # diffuse_direct_result = self.diffuse_decoder(diffuse_decoder_input) * light_albedo
        # B,_,_,C = diffuse_direct_result.shape
        # ma = diffuse_direct_result.reshape(B,-1,C).max(dim=1).values.unsqueeze(1).unsqueeze(1)
        # normalize_diffuse_direct_result = diffuse_direct_result / ma
        indirect_encoder_input = torch.cat([rsm_input[...,:64],light_albedo,light_position,light_normal],dim=-1)
        indirect_feature = self.indirect_encoder(indirect_encoder_input).unsqueeze(1).unsqueeze(1).repeat(1,512,512,1)
        indirect_decoder_input = torch.cat([indirect_feature,gbuffer_input],dim=-1)
        indirect_result = self.indirect_decoder(indirect_decoder_input) 
        indirect_result = torch.where(mask,gt,indirect_result)
        return indirect_result
class OnlyDirectDecoder(nn.Module):
    def __init__(self, configs, loss_config):
        super().__init__()
        self.configs = configs
        self.angular_size = configs["light_angular_resolution"]
        self.space_size = configs["light_direction_resolution"]
        self.local_gbuffer = configs["local_gbuffer"]
        self.loss_config = loss_config
        self.channel_cut = True
        self.light_input_dim = 9
        if self.channel_cut:
            self.light_input_dim = 7
        self.principle_triplane_encoder = ChannelCutDiTransformerPlane(configs)
        self.global_light_encoder = FourDTransformer(
                                                        angular_size=self.angular_size ,
                                                        space_size=self.space_size ,
                                                        angular_block_size=1,
                                                        space_block_size=16,
                                                        num_layers=configs["attention_layer_cnt"],
                                                        num_heads=16,
                                                        input_dim=self.light_input_dim,
                                                        hidden_dim=configs["plane"]["encoder_direct_dim"],
                                                        mlp_dim=configs["plane"]["encoder_direct_dim"] * 4,
                                                        output_dim=configs["plane"]["encoder_direct_dim"],
                                                        linear_output=True
                                                     )
        self.specular_feature = configs["plane"]["decoder_direct_dim"]

        self.loss_func = Loss(loss_config) if loss_config else None
        specular_feature = self.specular_feature
        self.output_dim = 3
        self.conditional_layer = False
        if self.channel_cut:
            self.output_dim = 1
        if self.local_gbuffer:
            self.diffuse_decoder = nn.Sequential(
                fc_layer(self.specular_feature * 1 + 12, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.output_dim))
            self.specular_decoder = nn.Sequential(
                fc_layer(self.specular_feature * 1 + 12, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.specular_feature * 1),
                fc_layer(self.specular_feature * 1, self.output_dim))

        self.dir_data = self.read_dir(configs["light_path"])
        self.dir_data = nn.Parameter(torch.Tensor(self.dir_data / \
                                                  numpy.linalg.norm(self.dir_data, axis=-1)[
                                                      ..., numpy.newaxis]).unsqueeze(0), requires_grad=False)
        if configs["temporal"]:
            self.shadow_network = TemporalShadowNetwork(16 + 4 + 1 + 1 + 16 + 4 + 1 + 1)
        else:
            self.shadow_network = SmallKernelShadowNetwork(16 + 4 + 1 )
        self.direct_compress_layer = nn.Linear(self.specular_feature, 16)
        self.indirect_feature =256
        self.indirect_proxy = IndirectFarwardProxy(self.indirect_feature,self.diffuse_decoder)
        self.sampling_way = configs["sampling"]
        # volume test
        # self.scene_compress_encoder = nn.Sequential(
        #     fc_layer(512 ,128)
        # )
        # self.scene_line_encoder = nn.Sequential(
        #     nn.Linear(64 * 32,1024)
        # )
        # self.scene_block_encoder = nn.Sequential(
        #     fc_layer(14 * 64, 512),
        #     fc_layer(512,512),
        #     fc_layer(512,512)
        # )
        # self.direct_decoder = nn.Sequential(
        #         fc_layer(self.indirect_feature * 1 + 12,self.indirect_feature),
        #         fc_layer(self.indirect_feature,self.indirect_feature),
        #         fc_layer(self.indirect_feature,self.indirect_feature),
        #         fc_layer(self.indirect_feature, 1))
        # self.light_to_block =  nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(self.indirect_feature, self.indirect_feature * 2, bias=True)
        # )
    
        # self.scene_transformer_encoder =BasicTransformer(512,16,512 * 4,dropout= 0)
        
        #self.voxel_input = ["voxelToLight","voxelPosition","voxelNormal","voxelDiffuse","voxelSpecular","voxelRoughness","voxelFresnel","voxelShadowMask"]
        

    def read_dir(self, dir_path):
        dir_fn = dir_path
        with open(dir_fn, 'rb') as gf:
            dctx = zstd.ZstdDecompressor()
            dir_data = pickle.loads(dctx.decompress(gf.read()))
        gf.close()
        #print(dir_data.shape)
        return dir_data

    
    def get_lightformer_input(self, localLightData, localData,channel_cnt):
        z_f = localLightData['shadow']['pixel_emitter_distance'] 
        z = localLightData['shadow']['occluder_emitter_distance'] 

        position_mask = localData["position_mask"]
        z_f[position_mask[..., 0]] = 3
        z[position_mask[..., 0]] = -3
        normal = localData["gbuffer"]["normal"]

        toLight = localData["toLight"][..., :3]
        toView = localData["gbuffer"]["view_dir"]
        c_c = torch.sum(localData["gbuffer"]['normal'] * localData["gbuffer"]['view_dir'], axis=-1)[
            ..., None]  # diffuse # TODO: Use this feature?
        depth = localData["gbuffer"]["depth"]
        #depth =normalize_depth_with_mask(depth,position_mask[...,0:1])
        localData["lightformer_input0"] = z_f - z
        localData["lightformer_input1"] = z_f / z
        localData["lightformer_input2"] = depth[:1,...]
        localData["lightformer_input3"] =  c_c[:1,...]
        localData["normal_dot_view"] = c_c
        shadow_tanh = localData["tanh_shadow"]
        localData["hard_shadow"] = torch.where(z_f - z > (0.0009 + localData["tanh_shadow"] * 0.003), 0, 1)
        return torch.cat([z_f - z, z_f / z, depth[:,...], c_c[:,...]], dim=-1)

    def down_sampleing_plane(self, light_input, angular_size, space_size):
        B, W1, H1, W2, H2, C = light_input.shape
        angular_pool = nn.AvgPool2d(angular_size)
        space_pool = nn.AvgPool2d(space_size)
        light_input = space_pool(light_input.permute(0, 1, 2, 5, 3, 4).reshape(B 
        * W1 * H1, C, W2, H2))
        W2, H2 = W2 // space_size, H2 // space_size
        light_input = light_input.reshape(B, W1, H1, C, W2, H2).permute(0, 4, 5, 3, 1, 2).reshape(B * W2 * H2, C, W1,
                                                                                                  H1)
        light_input = angular_pool(light_input)
        W1, H1 = W1 // angular_size, H1 // angular_size
        light_input = light_input.reshape(B, W2, H2, C, W1, H1).permute(0, 4, 5, 1, 2, 3)
        return light_input
    

    def forward_light(self,data,channel_cnt=3):

        data["direction"] = self.dir_data
        lightData = data
        
        light_input = torch.cat(
            [lightData["radiance"],lightData["direction"].repeat(lightData["radiance"].shape[0],1,1,1,1,1),lightData["position"]], dim=-1)
        #print(light_input.shape)
        if torch.any(light_input.isnan()):
            print("light input nan")
            exit()
        global_light_feature = self.global_light_encoder(light_input)
        if torch.any(global_light_feature.isnan()):
            print("global_light_feature nan")
            exit()
        return self.principle_triplane_encoder.forward_triplane(global_light_feature)

    def buffer_process(self,data,forward_indirect):
        specular_reflect_dir = torch.sum(data["local"]["gbuffer"]["view_dir"] * data["local"]["gbuffer"]["normal"],
                                         dim=-1).unsqueeze(-1) * data["local"]["gbuffer"]["normal"] * 2 - \
                               data["local"]["gbuffer"]["view_dir"]
        data["local"]["gbuffer"]["specular_ray"] = specular_reflect_dir
        data["local"]["gbuffer"]["half_vec"] = normalize(
            (normalize(-data["local"]["gbuffer"]["lposition"]) + data["local"]["gbuffer"]["view_dir"]) / 2)
        data["local"]["gbuffer"]["dot"] = torch.sum(
            data["local"]["gbuffer"]["half_vec"] * data["local"]["gbuffer"]["normal"], dim=-1, keepdim=True)
        
        light_space_gbuffer = get_light_space_gbuffer(data,forward_indirect)
        return light_space_gbuffer

    def forward_direct(self,decoder_type,data,result,light_feature,light_space_gbuffer,shading,mask,channel_cnt=3):
        
        if self.local_gbuffer:
            specular_input = torch.cat(
                [light_feature, light_space_gbuffer["normal"], light_space_gbuffer["half_vec"],
                 light_space_gbuffer["specular_ray"],
                 data["local"]["gbuffer"]["roughness"],
                 data["local"]["gbuffer"]["dot"],
                 data["local"]["gbuffer"]["lenth"]], dim=-1)
            # specular_input = torch.cat(
            #     [light_feature, data["local"]["gbuffer"]["position"], light_space_gbuffer["normal"],
            #      light_space_gbuffer["view_dir"],
            #      data["local"]["gbuffer"]["roughness"],
            #      data["local"]["gbuffer"]["dot"],
            #      data["local"]["gbuffer"]["lenth"]], dim=-1)
  
        if decoder_type=="diffuse":
            direct_shading = self.diffuse_decoder(specular_input)
        elif decoder_type=="specular":
            direct_shading = self.specular_decoder(specular_input)
        elif decoder_type=="direct":
            direct_shading = self.direct_decoder(specular_input)
        # print(mask.shape)
        # print(shading.shape)
        # print(direct_shading.shape)
        direct_shading = torch.where(mask,
                                     shading,direct_shading)
        return direct_shading

    def forward_shadow(self,data,result,light_feature,light_space_gbuffer,channel_cnt=3):
        localData = data["local"]
        data["local"]["shadow_light_repr"] = self.direct_compress_layer(data["local"]["direct_light_reprs"])
        data["local"]["shadow_input"] = self.get_lightformer_input(data["local"]["lights"], data["local"],channel_cnt)
        shadow_result, weights, shadow_list,shadow_loss = self.shadow_network.step(data)
        shadow_result = shadow_result
        if torch.any(torch.isnan(shadow_result)):
            print("nan")
            exit()
        shadow_result = torch.where(
            data["local"]["mask"],
            localData["shadow"],
            shadow_result)
        
        for i in range(5):
            data["local"]["single_shadow_{}".format(i)] = shadow_list[i].permute(0,2,3,1)
        data["local"]["single_shadow_5"] = weights[...,-1:]
        # localData["shadow_weight0"] = weights[..., :3]
        # localData["shadow_weight1"] = weights[..., 3:]
        return shadow_result,shadow_loss

    def normalize_light(self,data):
        indirect_data = data["local"]["lights"]["shadow"]
        B,_,_,_ = indirect_data["light_position"].shape
        indirect_data = data["local"]["lights"]["shadow"]
        scene_bound_min = indirect_data["light_position"].reshape(B,-1,3).min(dim=1).values.unsqueeze(1).unsqueeze(1)
        scene_bound_max = indirect_data["light_position"].reshape(B,-1,3).max(dim=1).values.unsqueeze(1).unsqueeze(1)
        # print(scene_bound_min)
        # print(scene_bound_max)
        scale = 1/ (scene_bound_max - scene_bound_min)
        
        indirect_data["light_position"] = (indirect_data["light_position"] - scene_bound_min) / (scene_bound_max-scene_bound_min)
        indirect_data["light_normal"] = normalize((indirect_data["light_normal"] + 1e-4) * scale)
        data["local"]["gbuffer"]["normal"] = normalize((data["local"]["gbuffer"]["normal"] + 1e-4) * scale)
        data["local"]["gbuffer"]["half_vec"] = normalize((data["local"]["gbuffer"]["half_vec"] + 1e-4) * scale)
        data["local"]["gbuffer"]["view_dir"] = normalize((data["local"]["gbuffer"]["view_dir"] + 1e-4) * scale)
        data["local"]["gbuffer"]["lposition"] = normalize((data["local"]["gbuffer"]["lposition"] -scene_bound_min) * scale)
    
    # def forward_indirct(self,data,principle_tri_plane,light_space_gbuffer,gt,mask,channel_cnt=3):
    #     indirect_data = data["local"]["lights"]["shadow"]
    #     indirect_feature,indirect_coord = self.principle_triplane_encoder.fetch_normal_triplane(principle_tri_plane,indirect_data["light_position"])

    #     lightview_direct_input = torch.cat([indirect_feature,indirect_data["light_lnormal"],indirect_data["light_half_vec"],
    #                                         indirect_data["light_specular_ray"],indirect_data["light_roughness"],indirect_data["light_dot"],
    #                                         indirect_data["light_depth"]],dim=-1)
    #     result = self.diffuse_decoder(lightview_direct_input) *indirect_data["light_albedo"]
    #     data["local"]["lights"]["shadow"]["direct_shading"] = result
    #     light_flux = result
    #     B,_,_,C = light_flux.shape
    #     ma = light_flux.reshape(B,-1,C)
    #     ma = ma.max(dim=1).values 
    #     #print(ma.shape)
    #     ma = ma.unsqueeze(1).unsqueeze(1)
    #     light_flux = light_flux / ma
       
    #     data["local"]["lights"]["shadow"]["direct_shading"] = light_flux
    #     #self.normalize_light(data)
    #     indirect_light_input = torch.cat([light_flux,indirect_data["light_position"],indirect_data["light_normal"]],dim=-1)
    #     _,w,h,c = indirect_light_input.shape
    #     #print(indirect_light_input.shape)
    #     if self.cnn_encoder:
    #         indirect_representation = self.indirect_encoder(indirect_light_input).unsqueeze(1).unsqueeze(1).repeat(1,512,512,1)
    #     else:
    #         indirect_representation = self.indirect_encoder(indirect_light_input).reshape(_,-1,self.indirect_feature).mean(dim=1).unsqueeze(1).unsqueeze(1).repeat(1,512,512,1)
    #     #print(indirect_representation.shape)
    #     indirect_input = torch.cat([indirect_representation,data["local"]["gbuffer"]["normal"], data["local"]["gbuffer"]["half_vec"],
    #             data["local"]["gbuffer"]["view_dir"],
    #             data["local"]["gbuffer"]["roughness"],
    #             data["local"]["gbuffer"]["lposition"]],dim=-1)
    #     indirect_result = self.indirect_decoder(indirect_input)
    #     indirect_result = indirect_result * ma
    #     indirect_result = torch.where(mask,gt,indirect_result)
        
    #     #exit()
    #     return indirect_result
    
    def save_voxel(self,voxelData,name):

        pyexr.write("./{}.exr".format(name),voxelData.cpu().numpy()[0])


    def make_batched_voxel_grid(self,bounds, resolution=(32, 32, 32), device="cuda"):
        """
        Args:
            bounds: (B, 2, 3) tensor, 每个 batch 的 [min, max] 坐标
            resolution: (nx, ny, nz)
        Returns:
            voxel_grid: (B, nx, ny, nz, 3)  每个体素中心坐标
        """
        B = bounds.shape[0]
        nx, ny, nz = resolution
        min_xyz = bounds[:, 0, :]  # (B, 3)
        max_xyz = bounds[:, 1, :]  # (B, 3)

        # 计算每个轴的中心偏移（避免采到边界）
        xs = torch.linspace(0.5, nx - 0.5, nx, device=device) / nx
        ys = torch.linspace(0.5, ny - 0.5, ny, device=device) / ny
        zs = torch.linspace(0.5, nz - 0.5, nz, device=device) / nz
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")  # (nz, ny, nx)
        base_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (nz, ny, nx, 3)
        base_grid = base_grid.unsqueeze(0).to(device)  # (1, nz, ny, nx, 3)

        # 按 batch 缩放/平移到各自 bounds
        voxel_grid = base_grid * (max_xyz - min_xyz).view(B, 1, 1, 1, 3) + min_xyz.view(B, 1, 1, 1, 3)

        # 转换为 (B, nx, ny, nz, 3)
        voxel_grid = voxel_grid.permute(0, 3, 2, 1, 4)
        return voxel_grid

    def down_triplane(self,avg_voxel_position):
        avg_voxel_position = avg_voxel_position.reshape(1,32,32,32,3)
        position_xy = avg_voxel_position[:,:,:,0, :2].reshape(1,32,32,-1)
        position_xz = avg_voxel_position[:,:,0,:,[0,1]].reshape(1,32,32,-1)
        position_yz = avg_voxel_position[:,0,:,:,1:].reshape(1,32,32,-1)



    def volume_scene_encoding(self,data,principle_tri_plane):
        
        indirect_data = data["local"]["lights"]["shadow"]
        #print(data["local"]["voxelPosition"].shape)
        face,uv = relative_pos_to_cubemap_uv(data["local"]["voxelPosition"].reshape(-1,3))
        uv[...,1:2] = uv[...,1:2] / 6 + face.unsqueeze(-1) / 6
        b = data["local"]["voxelPosition"].shape[0]
        uv = uv.reshape(b,128 * 128,128,2)
        
        voxel_occ_depth = F.grid_sample(indirect_data["light_view_distance"].permute(0,3,1,2),uv* 2 -1,align_corners=False,mode="bilinear").permute(0,2,3,1)
        voxel_depth = data["local"]["voxelPosition"].norm(dim=-1,keepdim=True) 
      
        #pyexr.write("./occluder_gt.exr", indirect_data["occluder_emitter_distance"][0].cpu().numpy())
        #print("voxel Mask",data["local"]["voxelMask"].shape)
        #print("voxel depth",voxel_depth.shape)
        voxel_depth = torch.where(data["local"]["voxelMask"],0,voxel_depth)
        voxel_occ_depth = torch.where(data["local"]["voxelMask"],0,voxel_occ_depth)
        data["local"]["voxelShadowMask"] = torch.where(voxel_depth - voxel_occ_depth > 0.1,1,0 )
        voxel_input = []
        for key in self.voxel_input:
            voxel_input.append(data["local"][key].reshape(b,128,128,128,-1))
            #self.save_voxel(data["local"][key],"voxel_{}".format(key))
        voxel_input = torch.cat(voxel_input,dim=-1)
        #print("voxel_input",voxel_input.shape)
        voxel_mask = data["local"]["voxelMask"].reshape(b,128,128,128,1)
        voxel_input = torch.where(voxel_mask,-10,voxel_input)
        voxel_mask_value = torch.where(voxel_mask,0,1)
        #print("voxel_mask",voxel_mask.shape)
        # determine how to sampling light, caculate weighted mean of position and use it as light sampling weight
        voxel_mask = voxel_mask_value.reshape(b,32,4,32,4,32,4,1).permute(0,1,3,5,2,4,6,-1).reshape(b,32,32,32,-1).sum(dim=-1,keepdim=True)
        voxel_position = ((data["local"]["voxelPosition"].reshape(b,128,128,128,3) * voxel_mask_value).reshape(b,32,4,32,4,32,4,3).permute(0,1,3,5,2,4,6,-1).reshape(b,32,32,32,64,3) ).sum(dim=-2) 
        avg_voxel_position = voxel_position / (voxel_mask + 1e-4)
        data["local"]["avg_voxel_position"] = avg_voxel_position
        pos = data["local"]["voxelPosition"].reshape(b, 128*128*128, 3)

        # 计算最小 / 最大值
        min_xyz = pos.min(dim=1).values  # (B,3)
        max_xyz = pos.max(dim=1).values  # (B,3)

        # 组合
        volume_scope = torch.stack([min_xyz, max_xyz], dim=1)  # (B,2,3)

        data["local"]["volume_scope"] = volume_scope
        #print("volume_scope",volume_scope)
        generate_voxel_position = self.make_batched_voxel_grid(data["local"]["volume_scope"] - data["local"]["light_pos"].unsqueeze(1)).permute(0,3,2,1,4)
        #print("generate_voxel_position",generate_voxel_position.shape)
 
        #light_xy,light_xz,light_yz,indirect_coord = self.principle_triplane_encoder.fetch_normal_triplane_byvolume(principle_tri_plane,avg_voxel_position.reshape(1,1024,32,3))
        light_feature,indirect_coord = self.principle_triplane_encoder.fetch_normal_triplane(principle_tri_plane,generate_voxel_position.reshape(b,1024,32,3))
        light_feature = light_feature.reshape(b,32,32,32,-1)
        # pyexr.write("./avg_voxel_position.exr",avg_voxel_position[0].reshape(1024,32,3).cpu().numpy())
        # pyexr.write("./voxel_mask.exr",voxel_mask[0].reshape(1024,32,1).cpu().numpy())
        scene_feature = self.scene_block_encoder(voxel_input.reshape(b,32,4,32,4,32,4,-1).permute(0,1,3,5,2,4,6,7).reshape(b,32,32,32,-1))

        #print("scene_feature",scene_feature.shape)
        # plane_xy = self.scene_line_encoder(grid_feature.reshape(1,32,32,-1))
        # plane_xz = self.scene_line_encoder(grid_feature.permute(0,1,3,2,4).reshape(1,32,32,-1))
        # plane_yz = self.scene_line_encoder(grid_feature.permute(0,2,3,1,4).reshape(1,32,32,-1))
        scale,shift = self.light_to_block(scene_feature).chunk(2, dim=-1)
        # print("light_feature ",light_feature.shape)
        # print("scale ",scale.shape)
        light_info = light_feature + light_feature * scale + shift
        # scene_compress = self.scene_compress_encoder(torch.cat([scene_feature,light_info],dim=-1))
        # B = scene_feature.shape[0]
        # plane_xy = self.scene_line_encoder(scene_compress.reshape(B,32,32,-1))
        # plane_xz = self.scene_line_encoder(scene_compress.permute(0,1,3,2,4).reshape(B,32,32,-1))
        # plane_yz = self.scene_line_encoder(scene_compress.permute(0,2,3,1,4).reshape(B,32,32,-1))
        # B,W,H,C = plane_xy.shape
        # plane_xy = self.scene_transformer_encoder(plane_xy.reshape(B,-1,C))
        # plane_xz = self.scene_transformer_encoder(plane_xz.reshape(B,-1,C))
        # plane_yz = self.scene_transformer_encoder(plane_yz.reshape(B,-1,C))
        # print(plane_xy.shape)
        # exit()

        # print("scale",scale.shape)
        # print("light_feature",light_feature.shape)
        # light_xy = light_xy * scale_xy + shift_xy + light_xy
        # light_xz = light_xz * scale_xz + shift_xz + light_xz

        return light_info,scene_feature,generate_voxel_position


    def forward_test(self, compressed_principle_triplane,data, need_diffuse, need_specular, need_shadow, need_indirect, need_indirect_direct, channel_cnt):

        print(compressed_principle_triplane.shape)

        # =============================================================
        # light space gbuffer
        # =============================================================
        s, e = start_timer()
        light_space_gbuffer = self.buffer_process(data, need_indirect)


        for key in light_space_gbuffer:
            data["local"][key] = light_space_gbuffer[key]

        # =============================================================
        # tri-plane sampling
        # =============================================================
        s, e = start_timer()
        if self.sampling_way == "normal":
            light_feature, voxel_coord = self.principle_triplane_encoder.fetch_normal_triplane(
                compressed_principle_triplane, data["local"]["gbuffer"]["lposition"]
            )
        elif self.sampling_way == "traditional":
            light_feature, voxel_coord = self.principle_triplane_encoder.fetch_traditional_triplane(
                compressed_principle_triplane, data["local"]["gbuffer"]["lposition"]
            )
        elif self.sampling_way == "oct":
            light_feature, voxel_coord = self.principle_triplane_encoder.fetch_spherical_tri_plane(
                compressed_principle_triplane, data["local"]["gbuffer"]["lposition"]
            )
    

        # ====== 你的原代码保持不动 =======
        if torch.any(light_feature.isnan()):
            print("light_feature nan")
            exit()
        data["local"]["sampled_principle_value"] = light_feature[..., :3]
        data["local"]["direct_light_reprs"] = light_feature
        data["local"]["voxel_coord"] = voxel_coord

        result = {}

        # =============================================================
        # need_diffuse
        # =============================================================
        if need_diffuse:
            s, e = start_timer()
            pred_diffuse = self.forward_direct(
                "diffuse", data, result, light_feature, light_space_gbuffer,
                data["local"]["log1p_diffuse_direct_shading"],
                data["local"]["mask"] | data["local"]["albedo_mask"], channel_cnt
            )
    
            result["log1p_diffuse_direct_shading"] = pred_diffuse

        # =============================================================
        # need_specular
        # =============================================================
        if need_specular:
            s, e = start_timer()
            pred_specular = self.forward_direct(
                "specular", data, result, light_feature, light_space_gbuffer,
                data["local"]["log1p_specular_direct_shading"],
                data["local"]["mask"] | data["local"]["specular_mask"], channel_cnt
            )
            timers["specular"] = end_timer(s, e)
            result["log1p_specular_direct_shading"] = pred_specular

        # =============================================================
        # need_shadow
        # =============================================================
        if need_shadow:
            s, e = start_timer()
            result["shadow"], shadow_loss = self.forward_shadow(
                data, result, light_feature, light_space_gbuffer, channel_cnt
            )
            timers["shadow"] = end_timer(s, e)

        # =============================================================
        # loss compute
        # =============================================================

        loss_map = {}
        return data, result, loss_map, {}
        
    def forward(self, data, need_diffuse, need_specular, need_shadow, need_indirect, need_indirect_direct, channel_cnt):
        timers = {}   # <---- 用于存储每个阶段的耗时(ms)
        # result  = {}
        # light_space_gbuffer = self.buffer_process(data, need_indirect)
        # if need_indirect:
        #     s, e = start_timer()
        #     indirect_result = self.forward_indirct(
        #         data, None, light_space_gbuffer,
        #         torch.cat([data["local"]["log1p_diffuse_indirect_shading"],
        #                 data["local"]["log1p_specular_indirect_shading"]], dim=-1),
        #         torch.cat([data["local"]["mask"] | data["local"]["albedo_mask"],
        #                 data["local"]["mask"] | data["local"]["specular_mask"]], dim=-1),
        #         channel_cnt
        #     )
        #     timers["indirect"] = end_timer(s, e)

        #     result["log1p_diffuse_indirect_shading"] = indirect_result[...,:1]
        #     result["log1p_specular_indirect_shading"] = indirect_result[...,1:2]
        # if self.loss_func is not None:
        #     loss_map = self.loss_func(result, data)
        # else:
        #     loss_map = None

        # return data, result, loss_map, timers
        # =============================================================
        # timmer: principle triplane
        # =============================================================
        # print(data["global"]["compressed_principle_triplane"].shape)
        # exit()
        #s, e = start_timer()
        if "compressed_principle_triplane" in data["global"].keys():
            compressed_principle_triplane =data["global"]["compressed_principle_triplane"]
            #print(compressed_principle_triplane.shape)
        else:
            principle_tri_plane, compressed_principle_triplane = self.forward_light(data["global"], channel_cnt)
        # print(compressed_principle_triplane.shape)
        # for i in range(3):
        #     for j in range(16):
        #         data["local"]["{}_{}".format(i,j)] = compressed_principle_triplane[0,i][...,j*4:(j*4+4)].unsqueeze(0)
        #         print(data["local"]["{}_{}".format(i,j)].shape)
            
        #timers["forward_light"] = end_timer(s, e)

        # if torch.any(principle_tri_plane.isnan()):
        #     print("principle_triplane nan")
        #     exit()

        # =============================================================
        # light space gbuffer
        # =============================================================
        #s, e = start_timer()
        light_space_gbuffer = self.buffer_process(data, need_indirect)
        #timers["buffer_process"] = end_timer(s, e)

        for key in light_space_gbuffer:
            data["local"][key] = light_space_gbuffer[key]

        # =============================================================
        # tri-plane sampling
        # =============================================================
        #s, e = start_timer()
        if self.sampling_way == "normal":
            light_feature, voxel_coord = self.principle_triplane_encoder.fetch_normal_triplane(
                compressed_principle_triplane, data["local"]["gbuffer"]["lposition"]
            )
        elif self.sampling_way == "traditional":
            light_feature, voxel_coord = self.principle_triplane_encoder.fetch_traditional_triplane(
                compressed_principle_triplane, data["local"]["gbuffer"]["lposition"]
            )
        elif self.sampling_way == "oct":
            light_feature, voxel_coord = self.principle_triplane_encoder.fetch_spherical_tri_plane(
                compressed_principle_triplane, data["local"]["gbuffer"]["lposition"]
            )
        #timers["sample_triplane"] = end_timer(s, e)

        # ====== 你的原代码保持不动 =======
        if torch.any(light_feature.isnan()):
            print("light_feature nan")
            exit()
        data["local"]["sampled_principle_value"] = light_feature[..., :3]
        data["local"]["direct_light_reprs"] = light_feature
        data["local"]["voxel_coord"] = voxel_coord

        result = {}

        # =============================================================
        # need_indirect_direct
        # =============================================================
        #print(data["local"]["volume_scope"].shape)
        #exit()
        if need_indirect_direct:
            s, e = start_timer()
            light_info, plane_info, voxel_l_position = self.volume_scene_encoding(data, principle_tri_plane)

            gbuffer_volume_position = (data["local"]["gbuffer"]["position"] - 
                                    data["local"]["volume_scope"][:,0:1,:].unsqueeze(1)) / \
                                    (data["local"]["volume_scope"][:,1:2,:].unsqueeze(1) - 
                                    data["local"]["volume_scope"][:,0:1,:].unsqueeze(1))
            print(gbuffer_volume_position.shape)
            gbuffer_volume_position = gbuffer_volume_position * 2 - 1
            data["local"]["gbuffer_volume_position"] = (gbuffer_volume_position +1) /2
            indirect_light_feature = F.grid_sample(
                light_info.permute(0,4,1,2,3),
                gbuffer_volume_position.unsqueeze(1),
                mode="bilinear",
                align_corners=False
            ).squeeze(2)
 

            data["local"]["light_info_visual"] = indirect_light_feature[...,:3]
            result["log1p_direct_shadow_shading"] = self.forward_direct(
                "direct", data, result, indirect_light_feature, light_space_gbuffer,
                data["local"]["log1p_direct_shadow_shading"],
                data["local"]["mask"], channel_cnt
            )
            timers["indirect_direct"] = end_timer(s, e)

        # =============================================================
        # need_indirect
        # =============================================================
        if need_indirect:
            #s, e = start_timer()
            indirect_data = data["local"]["lights"]["shadow"]
            indirect_feature,indirect_coord = self.principle_triplane_encoder.fetch_normal_triplane(compressed_principle_triplane,indirect_data["light_position"])

            rsm_input = torch.cat([indirect_feature,indirect_data["light_lnormal"],indirect_data["light_half_vec"],
                                                indirect_data["light_specular_ray"],indirect_data["light_roughness"],indirect_data["light_dot"],
                                                indirect_data["light_depth"],indirect_data["light_albedo"],indirect_data["light_position"],indirect_data["light_normal"]],dim=-1)
            screen_input = torch.cat([data["local"]["gbuffer"]["normal"], data["local"]["gbuffer"]["half_vec"],
                data["local"]["gbuffer"]["view_dir"],
                data["local"]["gbuffer"]["roughness"],
                data["local"]["gbuffer"]["lposition"]],dim=-1)
            indirect_result,n_d = self.indirect_proxy.train_step(
                rsm_input,screen_input,torch.cat([data["local"]["log1p_diffuse_indirect_shading"],
                        data["local"]["log1p_specular_indirect_shading"]], dim=-1),
                torch.cat([data["local"]["mask"] | data["local"]["albedo_mask"],
                        data["local"]["mask"] | data["local"]["specular_mask"]], dim=-1)
            )
            data["local"]["lights"]["shadow"]["direct_shading"] = n_d
            #timers["indirect"] = end_timer(s, e)

            result["log1p_diffuse_indirect_shading"] = indirect_result[...,:1]
            result["log1p_specular_indirect_shading"] = indirect_result[...,1:2]

        # =============================================================
        # need_diffuse
        # =============================================================
        if need_diffuse:
            #s, e = start_timer()
            pred_diffuse = self.forward_direct(
                "diffuse", data, result, light_feature, light_space_gbuffer,
                data["local"]["log1p_diffuse_direct_shading"],
                data["local"]["mask"] | data["local"]["albedo_mask"], channel_cnt
            )
            #timers["diffuse"] = end_timer(s, e)
            result["log1p_diffuse_direct_shading"] = pred_diffuse

        # =============================================================
        # need_specular
        # =============================================================
        if need_specular:
            #s, e = start_timer()
            pred_specular = self.forward_direct(
                "specular", data, result, light_feature, light_space_gbuffer,
                data["local"]["log1p_specular_direct_shading"],
                data["local"]["mask"] | data["local"]["specular_mask"], channel_cnt
            )
            #timers["specular"] = end_timer(s, e)
            result["log1p_specular_direct_shading"] = pred_specular

        # =============================================================
        # need_shadow
        # =============================================================
        if need_shadow:
            #s, e = start_timer()
            result["shadow"], shadow_loss = self.forward_shadow(
                data, result, light_feature, light_space_gbuffer, channel_cnt
            )
            #timers["shadow"] = end_timer(s, e)

        # =============================================================
        # loss compute
        # =============================================================
        #s, e = start_timer()
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)
        else:
            loss_map = None
        #timers["loss"] = end_timer(s, e)
        print(timers)
        return data, result, loss_map, timers