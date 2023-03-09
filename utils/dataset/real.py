import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset
import json
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import math
from pathlib import Path


def normalize_v(x) -> np.ndarray:
    return x / np.linalg.norm(x)
def read_cam_params(camFile: Path) -> list:
    """ read open gl camera """
    assert camFile.exists()
    with open(str(camFile), 'r') as camIn:
        cam_data = camIn.read().splitlines()
    cam_num = int(cam_data[0])
    cam_params = np.array([x.split(' ') for x in cam_data[1:]]).astype(np.float32)
    assert cam_params.shape[0] == cam_num * 3
    cam_params = np.split(cam_params, cam_num, axis=0) # [[origin, lookat, up], ...]
    return cam_params

def open_exr(file,img_hw):
    """ open image exr file """
    img = cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[...,[2,1,0]]
    img = torch.from_numpy(img.astype(np.float32))
    return img


def get_direction(k,img_hw):
    """ get camera ray direction (unormzlied)
        k: 3x3 camera intrinsics
        img_hw: image height and width
    """
    screen_y,screen_x = torch.meshgrid(torch.linspace(0.5,img_hw[0]-0.5,img_hw[0]),
                                torch.linspace(0.5,img_hw[1]-0.5,img_hw[1]))
    rays_d = torch.stack([
        (screen_x-k[0,2])/k[0,0],
        (screen_y-k[1,2])/k[1,1],
        torch.ones_like(screen_y)
    ],-1).reshape(-1,3)
    return rays_d

def to_world(rays_d,c2w,ray_diff,k):
    """ world sapce camera ray origin and direction
    Args:
        rays_d: HWx3 unormalized camera ray direction (local)
        c2w: 3x4 camera to world matrix
        ray_diff: True if return ray differentials
        k: 3x3 camera intrinsics
    Return:
        HWx3 camera origin
        HWx3 camera direction (unormzlied) if ray_diff==True
        HWx3 dxdu if ray_diff==True
        HWx3 dydv if ray_diff==True
    """
    rays_x = c2w[:,3:].T*torch.ones_like(rays_d)
    rays_d = rays_d@c2w[:3,:3].T
    if ray_diff:
        dxdu = torch.tensor([1.0/k[0,0],0,0])[None].expand_as(rays_d)@c2w[:3,:3].T
        dydv = torch.tensor([0,1.0/k[1,1],0])[None].expand_as(rays_d)@c2w[:3,:3].T
        return rays_x,rays_d,dxdu,dydv
    else:
        return rays_x,NF.normalize(rays_d,dim=-1)
        


class RealDataset(Dataset):
    """ Real world capture dataset in structure:
    Scene/
        Image/{:03d}_0001.exr HDR images
        segmentation/{:03d}.exr semantic segmentations
        cam.txt Image extrinsics
        K_list.txt Image intrinsics
        scene.obj Scene mesh
    """
    def __init__(self, root_dir, split='train', pixel=True,ray_diff=False):
        """
        Args:
            root_dir: dataset root folder
            split: train or val
            pixel: whether load every camera pixel
            ray_diff: whether return ray differentials
        """
        self.root_dir = root_dir
        self.pixel=pixel
        self.split = split

        # find image hight x width
        self.img_hw = cv2.imread(os.path.join(root_dir,'Image/000_0001.exr'),-1).shape[:2]

        self.ray_diff = ray_diff
        

        C2Ws_raw = read_cam_params(Path(self.root_dir,'cam.txt'))
        C2Ws = []
        # convert opengl camera to mitsuba
        for i,c2w_raw in enumerate(C2Ws_raw):
            origin, lookat, up = np.split(c2w_raw.T, 3, axis=1)
            origin = origin.flatten()
            lookat = lookat.flatten()
            up = up.flatten()
            at_vector = normalize_v(lookat - origin)
            assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular

            t = origin.reshape((3, 1)).astype(np.float32)
            R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1).astype(np.float32)
            C2Ws.append(np.hstack((R, t)))
        Ks = read_cam_params(Path(self.root_dir,'K_list.txt'))
        
        C2Ws = np.stack(C2Ws,0)
        Ks = np.stack(Ks,0)
        C2Ws = torch.from_numpy(C2Ws).float()
        Ks = torch.from_numpy(Ks).float()
        
        self.C2Ws = C2Ws
        self.Ks = Ks
        
        if self.pixel:
            # load all camera pixels
            self.all_rays = []
            for idx in range(len(self.Ks)):
                k = self.Ks[idx]
                c2w = self.C2Ws[idx]
                img_hw = self.img_hw
                img = open_exr(Path(self.root_dir,     
                    'Image','{:03d}_0001.exr'.format(idx)),img_hw).reshape(-1,3).clamp_min(0)
                
                rays_d = get_direction(k,img_hw)
                
                if self.ray_diff:
                    # load ray differential
                    rays_x,rays_d,dxdu,dydv = to_world(rays_d,c2w,self.ray_diff,k)
                    self.all_rays.append(torch.cat([
                        rays_x,rays_d,dxdu,dydv,
                        img
                    ],-1))
                else:
                    rays_x,rays_d = to_world(rays_d,c2w,self.ray_diff,k)
                    self.all_rays.append(torch.cat([
                        rays_x,rays_d,
                        img
                    ],-1))
            self.all_rays = torch.cat(self.all_rays,0)
    
    def __len__(self,):
        if self.pixel == True:
            return len(self.all_rays)
        if self.split == 'val':
            # only load 8 images for validation of reconstruction
            return 8
        return len(self.C2Ws)
    
    def __getitem__(self,idx):
        if self.pixel:
            tmp = self.all_rays[idx]
            return {
                'rays': tmp[...,:-3],
                'rgbs': tmp[...,-3:],
            }
        k = self.Ks[idx]
        c2w = self.C2Ws[idx]
        img_hw = self.img_hw

        img = open_exr(Path(self.root_dir,     
            'Image','{:03d}_0001.exr'.format(idx)),img_hw).reshape(-1,3).clamp_min(0)
        rays_d = get_direction(k,img_hw)
        
        if self.ray_diff:
            rays_x,rays_d,dxdu,dydv = to_world(rays_d,c2w,self.ray_diff,k)
            rays = torch.cat([
                rays_x,rays_d,dxdu,dydv
            ],-1)
        else:
            rays_x,rays_d = to_world(rays_d,c2w,self.ray_diff,k)
            rays = torch.cat([
                rays_x,rays_d,
            ],-1)
        return {
            'rays': rays,
            'rgbs': img,
            'c2w': c2w,
            'img_hw': img_hw,
        }
    
    
    
class InvRealDataset(Dataset):
    """ Real world capture dataset with diffuse and specular shadings
    Shading folder in structure:
    Scene/
        diffuse/{:03d}.exr diffuse shadings (L_d)
        specular/{:03d}_i_j.exr specular shadings (L_s^i(\sigma_j))
    """
    def __init__(self, root_dir, cache_dir,batch_size=None,split='train', pixel=True):
        """
        Args:
            root_dir: dataset root folder
            cache_dir: shadings folder
            split: train or val
            pixel: whether load every camera pixel
            batch_size: size of each ray batch if pixel==True
        """
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.pixel=pixel
        self.split = split
        self.ray_diff = True

        self.img_hw = cv2.imread(os.path.join(root_dir,'Image/000_0001.exr'),-1).shape[:2]
        self.batch_size = batch_size
        # approximate roughness channel by interpolating 6 samples
        self.roughness_level = 6
        
        C2Ws_raw = read_cam_params(Path(self.root_dir,'cam.txt'))
        C2Ws = []
        for i,c2w_raw in enumerate(C2Ws_raw):
            origin, lookat, up = np.split(c2w_raw.T, 3, axis=1)
            origin = origin.flatten()
            lookat = lookat.flatten()
            up = up.flatten()
            at_vector = normalize_v(lookat - origin)
            assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular

            t = origin.reshape((3, 1)).astype(np.float32)
            R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1).astype(np.float32)
            C2Ws.append(np.hstack((R, t)))
        Ks = read_cam_params(Path(self.root_dir,'K_list.txt'))
        
        C2Ws = np.stack(C2Ws,0)
        Ks = np.stack(Ks,0)
        C2Ws = torch.from_numpy(C2Ws).float()
        Ks = torch.from_numpy(Ks).float()
        
        self.C2Ws = C2Ws
        self.Ks = Ks
        
        if self.pixel:
            self.all_rays = []
            for idx in range(len(self.Ks)):
                k = self.Ks[idx]
                c2w = self.C2Ws[idx]
                img_hw = self.img_hw
                img = open_exr(Path(self.root_dir,     
                    'Image','{:03d}_0001.exr'.format(idx)),img_hw).reshape(-1,3).clamp_min(0)

                # load diffuse and specular shadings
                diffuse = open_exr(Path(self.cache_dir,'diffuse','{:03d}.exr'.format(idx)),img_hw).reshape(-1,3)
                speculars0,speculars1 = [],[]
                for r_idx in range(self.roughness_level):
                    specular0 = open_exr(Path(self.cache_dir,'specular','{:03d}_0_{}.exr'.format(idx,r_idx)),img_hw)
                    specular0 = specular0.reshape(-1,3)
                    speculars0.append(specular0)
                    specular1 = open_exr(Path(self.cache_dir,'specular','{:03d}_1_{}.exr'.format(idx,r_idx)),img_hw)
                    specular1 = specular1.reshape(-1,3)
                    speculars1.append(specular1)
                speculars0 = torch.cat(speculars0,-1)
                speculars1 = torch.cat(speculars1,-1)
                
                # segmentation mask
                segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'segmentation','{:03d}.exr'.format(idx)),-1))[...,0].reshape(-1,1).float()
                
                rays_d = get_direction(k,img_hw)
                rays_x,rays_d,dxdu,dydv = to_world(rays_d,c2w,self.ray_diff,k)
                self.all_rays.append(torch.cat([
                    rays_x,rays_d ,dxdu ,dydv ,
                    diffuse ,speculars0 ,speculars1 ,
                    segmentation ,
                    img 
                ],-1))
            self.all_rays = torch.cat(self.all_rays,0)

            # number of pixel batches
            self.batch_num = math.ceil(len(self.all_rays)*1.0/self.batch_size)
            self.idxs = torch.randperm(len(self.all_rays))
    
    def resample(self,):
        """ resample pixel batch """
        self.idxs = torch.randperm(len(self.all_rays))
    
    def __len__(self,):
        if self.pixel == True:
            return self.batch_num
        if self.split == 'val':
            return 8
        return len(self.C2Ws)
    
    def __getitem__(self,idx):
        if self.pixel:
            # find pixel indices for current batch
            b0 = idx*self.batch_size
            b1 = min(b0+self.batch_size,len(self.all_rays))
            
            idx = self.idxs[b0:b1]
            tmp = self.all_rays[idx]
            return {
                'rays': tmp[...,:12],
                'diffuse': tmp[...,12:15],
                'specular0': tmp[...,15:33].reshape(b1-b0,-1,3),
                'specular1': tmp[...,33:51].reshape(b1-b0,-1,3),
                'segmentation': tmp[...,51],
                'rgbs': tmp[...,52:55]
            }
        
        k = self.Ks[idx]
        c2w = self.C2Ws[idx]
        img_hw = self.img_hw
        img = open_exr(Path(self.root_dir,     
            'Image','{:03d}_0001.exr'.format(idx)),img_hw).reshape(-1,3).clamp_min(0)
        
        diffuse = open_exr(Path(self.cache_dir,'diffuse','{:03d}.exr'.format(idx)),img_hw).reshape(-1,3)
        speculars0,speculars1 = [],[]
        for r_idx in range(self.roughness_level):
            specular0 = open_exr(Path(self.cache_dir,'specular','{:03d}_0_{}.exr'.format(idx,r_idx)),img_hw)
            specular0 = specular0.reshape(-1,3)
            speculars0.append(specular0)
            specular1 = open_exr(Path(self.cache_dir,'specular','{:03d}_1_{}.exr'.format(idx,r_idx)),img_hw)
            specular1 = specular1.reshape(-1,3)
            speculars1.append(specular1)
        speculars0 = torch.stack(speculars0,-2)
        speculars1 = torch.stack(speculars1,-2)
        
        segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'segmentation','{:03d}.exr'.format(idx)),-1))[...,0].reshape(-1,1).float()
        
        
        rays_d = get_direction(k,img_hw)
        
        rays_x,rays_d,dxdu,dydv = to_world(rays_d,c2w,self.ray_diff,k)
        rays = torch.cat([
            rays_x,rays_d,dxdu,dydv
        ],-1)
     
        return {
            'rays': rays,
            'rgbs': img,
            'c2w': c2w,
            'diffuse':diffuse,
            'specular0': speculars0,
            'specular1': speculars1,
            'segmentation': segmentation,
            'img_hw': img_hw,
        }