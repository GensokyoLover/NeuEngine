import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import imageio.v3 as iio
import random
import pyexr
def load_exr_image(path, size=256):
    img = pyexr.read(path)  # (H,W,C)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    img = torch.from_numpy(img).permute(2,0,1)  # (3,H,W)

    resize = T.Resize((size, size))
    img = resize(img)
    return img   # (3,size,size)


def extract_patches(img, ys, xs, P):
    """
    img: (3,H,W) CPU tensor
    ys, xs: (K,) patch top-left positions
    return: (K,3,P,P)
    """
    K = ys.shape[0]
    H, W = img.shape[1], img.shape[2]

    # (K, P, P) absolute coordinate for each pixel in patch
    dy = torch.arange(P).view(1, P, 1)
    dx = torch.arange(P).view(1, 1, P)

    yy = ys.view(K,1,1) + dy    # (K,P,1)
    xx = xs.view(K,1,1) + dx    # (K,1,P)

    yy = yy.expand(-1, -1, P)   # (K,P,P)
    xx = xx.expand(-1, P, -1)   # (K,P,P)

    # flatten for gather
    coords = yy * W + xx  # (K,P,P)
    coords = coords.reshape(K, -1)   # (K, P*P)

    img_flat = img.reshape(3, -1)    # (3, H*W)
    img_flat = img_flat.unsqueeze(0).expand(K, -1, -1)  # (K,3,H*W)

    patches = torch.gather(img_flat, 2, coords.unsqueeze(1).expand(K,3,-1))
    return patches.reshape(K, 3, P, P)

class EXRBatchPatchDataset(Dataset):
    def __init__(self, folder, img_size=256, patch_size=16, num_patches=2000):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".exr")])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        print(f"Total images: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def load_exr_image(self, path):
        img = pyexr.read(path)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = torch.from_numpy(img).permute(2,0,1)  # (3,H,W)
        img = T.Resize((self.img_size, self.img_size))(img)
        
        return img

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.folder, fname)

        img = self.load_exr_image(img_path)  # (3,H,W)

        P = self.patch_size
        H = W = self.img_size
        Ny = H - P + 1
        Nx = W - P + 1

        # === vectorized unfold for entire image ===
        u = img.unfold(1, P, 1).unfold(2, P, 1)   # (3,Ny,Nx,P,P)
        u = u.permute(1,2,0,3,4).contiguous()     # (Ny,Nx,3,P,P)

        # === random sample K positions ===
        ys = torch.randint(0, Ny, (self.num_patches,))
        xs = torch.randint(0, Nx, (self.num_patches,))

        patches = u[ys, xs]   # <-- O(1) slicing, ultra fast

        pos = torch.stack([ys, xs], dim=1).float()  # (K,2)

        return img.unsqueeze(0), patches, pos
    

from torch.utils.data import Dataset
import zstandard as zstd
import pickle
class ImpostorTrainingDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 device="cuda"):

        """
        root_dir:
            e.g.  H:/Falcor/media/inv_rendering_scenes/bunny_ref_512_scale/
        
        training_keys:
            training_datasets_list (list of strings)
        
        buffer_channel:
            每个 key 的通道数，例如 {"raypos":3, "view":3, ...}
        
        transform:
            可选，对每个样本做数据增强（一般不用）
        
        device:
            "cuda" or "cpu"
        """

        self.root_dir = root_dir
        # self.training_keys = ["AccumulatePassoutput","albedo","depth","specular","normal","position","roughness","view","raypos","emission"]

        self.device = device
        self.root_dir = os.path.join(root_dir, "buckets_1")
        init_file_list = os.listdir(self.root_dir)
        self.roughness_floor = 1
        self.file_dict = {}
        for file in init_file_list:
            _,_,_,_,roughness,_ = file.split("_")
            roughness = int(roughness)
            if roughness not in self.file_dict.keys():
                self.file_dict[roughness] = []
            self.file_dict[roughness].append(file)
        # self.file_dict[0] = ["bucket_0_12_14_0_part0.pkl.zst"]
        if 0 not in self.file_dict.keys():
            self.file_dict[0] = []
        for i in range(1,5):
            if i not in self.file_dict.keys():
                self.file_dict[i] = []
            self.file_dict[i] = self.file_dict[i] + self.file_dict[i-1]
    def __len__(self):
        return len(self.file_dict[self.roughness_floor])

    def __getitem__(self, idx):
        file_list = self.file_dict[self.roughness_floor]
        idx = idx % len(file_list)
        scene_id = file_list[idx]

        folder = os.path.join(self.root_dir, f"{scene_id}")
        # read xxx.pkl.zst
        with open(folder, "rb") as f:
            compressed_data = f.read()
        decompressed_data = zstd.decompress(compressed_data)
        sample = pickle.loads(decompressed_data)
        

        for key in sample:
            sample[key] = sample[key].reshape(256, 256, *sample[key].shape[1:])

        return sample,scene_id
    
class ImpostorTrainingTestDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 device="cuda"):

        """
        root_dir:
            e.g.  H:/Falcor/media/inv_rendering_scenes/bunny_ref_512_scale/
        
        training_keys:
            training_datasets_list (list of strings)
        
        buffer_channel:
            每个 key 的通道数，例如 {"raypos":3, "view":3, ...}
        
        transform:
            可选，对每个样本做数据增强（一般不用）
        
        device:
            "cuda" or "cpu"
        """

        self.root_dir = root_dir
        # self.training_keys = ["AccumulatePassoutput","albedo","depth","specular","normal","position","roughness","view","raypos","emission"]

        self.device = device
        self.root_dir = os.path.join(root_dir, "test")
        init_file_list = os.listdir(self.root_dir)
        self.file_dict = []
        for file in init_file_list:
            self.file_dict.append(file)
        # self.file_dict[0] = ["bucket_0_12_14_0_part0.pkl.zst"]
        # for i in range(1,5):
        #     self.file_dict[i] = self.file_dict[i] + self.file_dict[i-1]
    def __len__(self):
        return len(self.file_dict)

    def __getitem__(self, idx):
        file_list = self.file_dict
        idx = idx % len(file_list)
        scene_id = file_list[idx]

        folder = os.path.join(self.root_dir, f"{scene_id}")
        # read xxx.pkl.zst
        with open(folder, "rb") as f:
            compressed_data = f.read()
        decompressed_data = zstd.decompress(compressed_data)
        sample = pickle.loads(decompressed_data)
        

        
        return sample,scene_id