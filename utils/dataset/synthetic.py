import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset
import json
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from PIL import Image
from torchvision import transforms as T
import cv2
import math


def get_ray_directions(H, W, focal):
    """ get camera ray direction
    Args:
        H,W: height and width
        focal: focal length
    """
    x_coords = torch.linspace(0.5, W - 0.5, W)
    y_coords = torch.linspace(0.5, H - 0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    directions = \
        torch.stack([-(i-W/2)/focal, -(j-H/2)/focal, torch.ones_like(i)], -1) 

    return directions

def get_rays(directions, c2w, focal=None):
    """ world space camera ray
    Args:
        directions: camera ray direction (local)
        c2w: 3x4 camera to world matrix
        focal: if not None, return ray differentials as well
    """
    R = c2w[:,:3]
    rays_d = directions @ R.T
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    if focal is not None:
        dxdu = torch.tensor([1.0/focal,0,0])[None,None].expand_as(directions)@R.T
        dydv = torch.tensor([0,1.0/focal,0])[None,None].expand_as(directions)@R.T
        dxdu = dxdu.view(-1,3)
        dydv = dydv.view(-1,3)
        return rays_o, rays_d, dxdu, dydv
    else:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d


def open_exr(file,img_hw):
    img = cv2.imread(file,cv2.IMREAD_UNCHANGED)[...,[2,1,0]]
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    #img = cv2.resize(img,img_hw,cv2.INTER_LANCZOS4)
    img = torch.from_numpy(img.astype(np.float32))
    return img

class SyntheticDataset(Dataset):
    """ synthetic dataset in structure:
    Scene/
        {SPLIT}/ train or val split
            Image/{:03d}_0001.exr HDR images
            Roughness/{:03d}_0001.exr Roughness
            DiffCol/{:03d}_0001.exr diffuse reflectance
            albedo/{:03d}.exr material reflectance a'=\int f d\omega_i
            Emit/{:03d}_0001.exr emission 
            IndexMA/{:03d}_0001.exr material part segmentation
            segmentation/{:03d}.exr semantic segmentation
            transforms.json c2w camera matrix file and fov
    """
    def __init__(self, root_dir, split='train', pixel=True, ray_diff=False):
        """
        Args:
            root_dir: dataset root folder
            split: train or val
            pixel: whether load every camera pixel
            ray_diff: whether load ray differentials
        """
        self.root_dir = os.path.join(root_dir,split) if split != 'relight'\
                      else os.path.join(root_dir,'val')
        self.pixel=pixel
        self.split = split

        self.img_hw = cv2.imread(os.path.join(root_dir,'train/Image/000_0001.exr'),-1).shape[:2]

        self.ray_diff = ray_diff
        
        with open(os.path.join(self.root_dir,
                               f"transforms.json"), 'r') as f:
            self.meta = json.load(f)

        # camera focal length and ray directions
        h,w = self.img_hw
        self.focal = (0.5*w/np.tan(0.5*self.meta['camera_angle_x'])).item()
        self.directions = \
            get_ray_directions(h, w, self.focal)
        
        # load every camera pixels
        if self.pixel:
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for cur_idx in range(len(self.meta['frames'])):
                frame = self.meta['frames'][cur_idx]
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)
                
                image_path = os.path.join(self.root_dir, 'Image','{:03d}_0001.exr'.format(cur_idx))
                img = open_exr(image_path,self.img_hw).reshape(-1,3)
                
                # load ground truth BRDF
                albedo = open_exr(os.path.join(self.root_dir,'DiffCol','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
                roughness = open_exr(os.path.join(self.root_dir,'Roughness','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)[...,:1]
                emission = open_exr(os.path.join(self.root_dir,'Emit','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
  
                self.all_rgbs += [img]
                
                if self.ray_diff==False:
                    rays_o, rays_d = get_rays(self.directions, c2w) 

                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                 albedo,
                                                 roughness,
                                                 emission
                                                ],1)]
                else:
                    rays_o,rays_d,dxdu,dydv = get_rays(self,directions,c2w,focal=self.focal)
                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                 dxdu,dydv,
                                                 albedo,
                                                 roughness,
                                                 emission
                                                ],1)]

            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)

    def __len__(self):
        if self.pixel==True:
            return len(self.all_rays)
        if self.split == 'val':
            # only show 8 images for reconstruction validation
            return 8
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.pixel: 
            tmp = self.all_rays[idx]
            if self.ray_diff == False:
                sample = {'rays': tmp[:8],
                          'rgbs': self.all_rgbs[idx],
                          'albedo': tmp[8:11],
                          'roughness': tmp[11],
                          'emission': tmp[12:15]
                         }
            else:
                sample = {'rays': tmp[:12],
                      'rgbs': self.all_rgbs[idx],
                      'albedo': tmp[12:15],
                      'roughness': tmp[15],
                      'emission': tmp[16:19]
                     }

        else:
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            cur_idx = idx
            
            image_path = os.path.join(self.root_dir, 'Image','{:03d}_0001.exr'.format(cur_idx))
            img = open_exr(image_path,self.img_hw).reshape(-1,3)
            
            albedo = open_exr(os.path.join(self.root_dir,'DiffCol','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
            roughness = open_exr(os.path.join(self.root_dir,'Roughness','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)[...,0]
            emission = open_exr(os.path.join(self.root_dir,'Emit','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
            
            if self.ray_diff == False:
                rays_o, rays_d = get_rays(self.directions, c2w)

                rays = torch.cat([rays_o, rays_d],1)
            else:
                rays_o, rays_d,dxdu,dydv = get_rays(self.directions, c2w,focal=self.focal)
                rays = torch.cat([rays_o, rays_d,dxdu,dydv],1)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'albedo':albedo,
                      'roughness': roughness,
                      'emission': emission
                     }

        return sample

class InvSyntheticDataset(Dataset):
    """ Synthetic dataset with diffuse and specular shadings
    Shading folder in structure:
    Scene/
        diffuse/{:03d}.exr diffuse shadings (L_d)
        specular/{:03d}_i_j.exr specular shadings (L_s^i(\sigma_j))
    """
    def __init__(self, root_dir, cache_dir, batch_size=None,split='train', pixel=True,has_part=False):
        """
        Args:
            root_dir: dataset root folder
            cache_dir: shadings folder
            split: train or val
            pixel: whether load every camera pixel
            batch_size: size of each ray batch if pixel==True
            has_part: whether use ground truth part segmentation or not (semantic segmentation)
        """
        self.root_dir = os.path.join(root_dir,split)
        self.cache_dir = cache_dir
        self.pixel=pixel
        self.split = split
        self.batch_size = batch_size
        self.has_part = has_part
        # approximate roughness channel by interpolating 6 samples
        self.roughness_level = 6
        
        self.img_hw = cv2.imread(os.path.join(root_dir,'train/Image/000_0001.exr'),-1).shape[:2]
        

        with open(os.path.join(self.root_dir,
                               f"transforms.json"), 'r') as f:
            self.meta = json.load(f)

        h,w = self.img_hw
        self.focal = (0.5*w/np.tan(0.5*self.meta['camera_angle_x'])).item()
        self.directions = get_ray_directions(h, w, self.focal)
            
        if self.pixel: 
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for cur_idx in range(len(self.meta['frames'])):
                frame = self.meta['frames'][cur_idx]
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)
                
                image_path = os.path.join(self.root_dir, 'Image','{:03d}_0001.exr'.format(cur_idx))
                img = open_exr(image_path,self.img_hw).reshape(-1,3)

                # ground truth brdf-emission
                albedo = open_exr(os.path.join(self.root_dir,'DiffCol','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
                roughness = open_exr(os.path.join(self.root_dir,'Roughness','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)[:,:1]
                emission = open_exr(os.path.join(self.root_dir,'Emit','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
                
                # load shadings
                diffuse = open_exr(os.path.join(self.cache_dir,'diffuse','{:03d}.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
                speculars0,speculars1 = [],[]
                for r_idx in range(self.roughness_level):
                    specular0 = open_exr(os.path.join(self.cache_dir,'specular','{:03d}_0_{}.exr'.format(cur_idx,r_idx)),self.img_hw).float()
                    specular0 = specular0.reshape(-1,3)
                    speculars0.append(specular0)
                    specular1 = open_exr(os.path.join(self.cache_dir,'specular','{:03d}_1_{}.exr'.format(cur_idx,r_idx)),self.img_hw).float()
                    specular1 = specular1.reshape(-1,3)
                    speculars1.append(specular1)
                speculars0 = torch.cat(speculars0,-1)
                speculars1 = torch.cat(speculars1,-1)
                
                # load part or semantic segmentation
                if self.has_part:
                    segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'IndexMA','{:03d}_0001.exr'.format(cur_idx)),-1))[...,0].reshape(-1,1).float()
                else:
                    segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'segmentation','{:03d}.exr'.format(cur_idx)),-1))[...,0].reshape(-1,1).float()

                self.all_rgbs += [img]
                rays_o, rays_d,dxdu,dydv = get_rays(self.directions, c2w, focal=self.focal) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             dxdu,
                                             dydv,
                                             albedo,
                                             roughness,
                                             diffuse,
                                             speculars0,
                                             speculars1,
                                             emission,
                                             segmentation
                                            ],1)] 
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            # number of camera ray batches
            self.batch_num = math.ceil(len(self.all_rays)*1.0/self.batch_size)
            self.idxs = torch.randperm(len(self.all_rays))
    
    def resample(self,):
        # resample camera ray batches
        self.idxs = torch.randperm(len(self.all_rays))

    def __len__(self):
        if self.pixel==True:
            return self.batch_num
        if self.split == 'val':
            return 8
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.pixel:
            b0 = idx*self.batch_size
            b1 = min(b0+self.batch_size,len(self.all_rays))
            
            # find camera ray indices in the batch
            idx = self.idxs[b0:b1]
            tmp = self.all_rays[idx]
            
            sample = {'rays': tmp[...,:12],
                      'albedo': tmp[...,12:15],
                      'roughness': tmp[...,15],
                      'diffuse': tmp[...,16:19],
                      'specular0': tmp[...,19:37].reshape(b1-b0,-1,3),
                      'specular1': tmp[...,37:55].reshape(b1-b0,-1,3),
                      'emission': tmp[...,55:58],
                      'segmentation': tmp[...,58],
                      'rgbs': self.all_rgbs[idx]}
            

        else:
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            cur_idx = idx
            
            image_path = os.path.join(self.root_dir, 'Image','{:03d}_0001.exr'.format(cur_idx))
            img = open_exr(image_path,self.img_hw).reshape(-1,3)
            
            albedo = open_exr(os.path.join(self.root_dir,'DiffCol','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
            roughness = open_exr(os.path.join(self.root_dir,'Roughness','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)[:,0]
            emission = open_exr(os.path.join(self.root_dir,'Emit','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
            
            diffuse = open_exr(os.path.join(self.cache_dir,'diffuse','{:03d}.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
            speculars0,speculars1 = [],[]
            for r_idx in range(self.roughness_level):
                specular0 = open_exr(os.path.join(self.cache_dir,'specular','{:03d}_0_{}.exr'.format(cur_idx,r_idx)),self.img_hw).float()
                specular0 = specular0.reshape(-1,3)
                speculars0.append(specular0)
                specular1 = open_exr(os.path.join(self.cache_dir,'specular','{:03d}_1_{}.exr'.format(cur_idx,r_idx)),self.img_hw).float()
                specular1 = specular1.reshape(-1,3)
                speculars1.append(specular1)
            speculars0 = torch.stack(speculars0,-2)
            speculars1 = torch.stack(speculars1,-2)
            
            if self.has_part:
                segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'IndexMA','{:03d}_0001.exr'.format(cur_idx)),-1))[...,0].reshape(-1).float()
            else:
                segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'segmentation','{:03d}.exr'.format(cur_idx)),-1))[...,0].reshape(-1).float()
            
            rays_o,rays_d,dxdu,dydv = get_rays(self.directions, c2w, focal=self.focal)

            rays = torch.cat([rays_o, rays_d,
                              dxdu,
                              dydv],-1)
        
            sample = {
                'rays': rays,
                'rgbs': img,
                'c2w': c2w,
                'albedo':albedo,
                'diffuse':diffuse,
                'specular0': speculars0,
                'specular1': speculars1,
                'roughness': roughness,
                'emission': emission,
                'segmentation': segmentation
            }
        return sample