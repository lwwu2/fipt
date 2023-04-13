import torch

import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

import math

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from utils.dataset import SyntheticDataset,RealDataset
from utils.path_tracing import ray_intersect
from model.slf import VoxelSLF
from model.brdf import BaseBRDF
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import time


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help='dataset folder')
    parser.add_argument('--output', type=str, required=True, help='output path')
    parser.add_argument('--dataset',type=str,required=True, help='dataset type')
    parser.add_argument('--voxel_num',type=int,default=256, help='resolution for voxel radiance cache')
    args = parser.parse_args()

    device = torch.device(0) # use gpu device 0

    DATASET_PATH = args.scene
    OUTPUT_PATH = args.output

    # load mesh
    mesh_path = os.path.join(DATASET_PATH,'scene.obj')
    assert Path(mesh_path).exists(), 'mesh not found: '+mesh_path
    
    scene = mitsuba.load_dict({
        'type': 'scene',
        'shape_id':{
            'type': 'obj',
            'filename': mesh_path, 
        }
    })

    # load dataset
    if args.dataset  == 'synthetic':
        dataset_fn = SyntheticDataset
    elif args.dataset == 'real':
        dataset_fn = RealDataset
    dataset = dataset_fn(DATASET_PATH,split='train',pixel=False)
    img_hw = dataset.img_hw
    
    os.makedirs(OUTPUT_PATH,exist_ok=True)


    start_time = time.time()
    # extract scene bounding box
    print('find scene bound')
    voxel_min = 1000.
    voxel_max = 0.0
    for idx in tqdm(range(len(dataset))):
        batch = dataset[idx]
        rays = batch['rays']
        xs = rays[...,:3]
        ds = rays[...,3:6]

        positions,_,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        if not valid.any():
            continue
        position = positions[valid]
        voxel_min = min(voxel_min,position.min())
        voxel_max = max(voxel_max,position.max())
    voxel_min = 1.1*voxel_min
    voxel_max = 1.1*voxel_max

    # find voxels that are not occupied
    print('find visible voxels')
    res_spatial = args.voxel_num
    SpatialHist = torch.zeros(res_spatial**3,device=device)
    for idx in tqdm(range(len(dataset))):
        batch = dataset[idx]
        rays = batch['rays']
        xs = rays[...,:3]
        ds = rays[...,3:6]

        positions,_,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        if not valid.any():
            continue
        
        position = (positions[valid]-voxel_min)/(voxel_max-voxel_min)
        position = (position*res_spatial).long().clamp(0,res_spatial-1)
        inds = position[...,0] + position[...,1]*res_spatial\
             + position[...,2]*res_spatial*res_spatial
        SpatialHist.scatter_add_(0,inds,torch.ones_like(inds).float())
    SpatialHist = SpatialHist.reshape(res_spatial,res_spatial,res_spatial)

    mask = (SpatialHist>0)
    
    # create voxle surface light field
    print('bake voxel surface light field')
    vslf = VoxelSLF(mask.cpu(),voxel_min.item(),voxel_max.item())
    vslf.radiance = torch.zeros_like(vslf.radiance)
    for idx in tqdm(range(len(dataset))):
        batch = dataset[idx]
        rays = batch['rays']
        radiance = batch['rgbs']
        xs = rays[...,:3]
        ds = rays[...,3:6]

        positions,_,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        if not valid.any():
            continue

        vslf.scatter_add(positions[valid].cpu(),radiance.to(device)[valid].cpu())

    # average pooling the radiance
    vslf.radiance = vslf.radiance/vslf.count[...,None].float().clamp_min(1)
    
    
    torch.save({
        'mask': (SpatialHist>0),
        'voxel_min': voxel_min.item(),
        'voxel_max': voxel_max.item(),
        'weight':vslf.state_dict()
    },os.path.join(OUTPUT_PATH,'vslf.npz'))

    for p in vslf.parameters():
        p.requires_grad=False
    vslf.to(device)
    material_net = BaseBRDF()

    denoiser = mitsuba.OptixDenoiser(img_hw[::-1])

    print('[bake_shading - init] time (s): ', time.time()-start_time)
    start_time = time.time()

    # bake diffuse shading
    print('bake diffuse')
    output_path = os.path.join(OUTPUT_PATH,'diffuse')
    os.makedirs(output_path,exist_ok=True)
    
    SPP = 256
    
    im_id = 0
    for batch in tqdm(dataset):
        rays = batch['rays']
        xs = rays[...,:3]
        ds = rays[...,3:6]

        positions,normals,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        position = positions[valid]
        normal = normals[valid]
        ds = ds.to(device)[valid]
        
        B = ds.shape[0]
        Ld_ = torch.zeros(B,3,device=device)
        batch_size = 10240*64//SPP

        # batched diffuse shading calculation
        for b in range(math.ceil(B*1.0/batch_size)):
            b0 = b*batch_size
            b1 = min(b0+batch_size,B)

            # importance sampling wi
            wi,_,_, = material_net.sample_diffuse(torch.rand((b1-b0)*SPP,2,device=device),
                                    normal[b0:b1].repeat_interleave(SPP,0))
    
            p_next,_,_,_,valid_next = ray_intersect(scene,
                        position[b0:b1].repeat_interleave(SPP,0)+mitsuba.math.RayEpsilon*wi,# prevent self intersection
                        wi.reshape(-1,3))
            
            # query surface light field
            Le = torch.zeros_like(wi)
            Le[valid_next] = vslf(p_next[valid_next])['rgb']

            Ld_[b0:b1] = Le.reshape(b1-b0,SPP,3).mean(1)

        # denoiser renderings
        Ld = torch.zeros_like(xs)
        Ld[valid.cpu()] = Ld_.cpu()
        Ld = Ld.reshape(*img_hw,3).numpy()
        Ld = denoiser(Ld).numpy()

        cv2.imwrite(os.path.join(output_path,'{:03d}.exr'.format(im_id)),Ld[:,:,[2,1,0]])
        im_id += 1
    
    print('[bake_shading - diffuse] time (s): ', time.time()-start_time)
    start_time = time.time()

    # bake specular shadings
    print('bake specular')
    output_path = os.path.join(OUTPUT_PATH,'specular')
    os.makedirs(output_path,exist_ok=True)


    SPPs = [64,128,128,128,128,128] # use different sampling rate
    im_id = 0

    # 6 roughness level
    roughness_level = torch.linspace(0.02,1.0,6)

    for batch in tqdm(dataset):
        rays = batch['rays']
        xs = rays[...,:3]
        ds = rays[...,3:6]
        
        positions,normals,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        position = positions[valid]
        normal = normals[valid]
        wo = -ds.to(device)[valid]

        B = position.shape[0]
        # caculate for each roughness value
        for r_idx,roughness in enumerate(roughness_level):
            SPP = SPPs[r_idx]
            Ls0_ = torch.zeros(B,3,device=device)
            Ls1_ = torch.zeros(B,3,device=device)
            
            # batched specular shading calculation
            batch_size = 10240*64//SPP
            for b in range(math.ceil(B*1.0/batch_size)):
                b0 = b*batch_size
                b1 = min(b0+batch_size,B)
                
                # importance sampling wi
                wi,_,g0,g1 = material_net.sample_specular(
                        torch.rand((b1-b0)*SPP,2,device=device),
                        wo[b0:b1].repeat_interleave(SPP,0),
                        normal[b0:b1].repeat_interleave(SPP,0),roughness
                )

                p_next,_,_,_,valid_next = ray_intersect(scene,
                        position[b0:b1].repeat_interleave(SPP,0)+mitsuba.math.RayEpsilon*wi,# prevent self intersection
                        wi.reshape(-1,3))
                
                # query surface light field
                Le = torch.zeros_like(wi)
                Le[valid_next] = vslf(p_next[valid_next])['rgb']

                Ls0_[b0:b1] = (Le*g0).reshape(b1-b0,SPP,3).mean(1)
                Ls1_[b0:b1] = (Le*g1).reshape(b1-b0,SPP,3).mean(1)

            Ls0 = torch.zeros_like(xs)
            Ls1 = torch.zeros_like(xs)
            Ls0[valid.cpu()] = Ls0_.cpu()
            Ls1[valid.cpu()] = Ls1_.cpu()
            
            Ls0 = Ls0.reshape(*img_hw,3).numpy()
            Ls1 = Ls1.reshape(*img_hw,3).numpy()

            if r_idx > 0: # no need for denoise of low roughness
                Ls0 = denoiser(Ls0).numpy()
                Ls1 = denoiser(Ls1).numpy()
                
            cv2.imwrite(os.path.join(output_path,'{:03d}_0_{}.exr'.format(im_id,r_idx)),Ls0[:,:,[2,1,0]])
            cv2.imwrite(os.path.join(output_path,'{:03d}_1_{}.exr'.format(im_id,r_idx)),Ls1[:,:,[2,1,0]])
        im_id += 1
    
    print('[bake_shading - specular] time (s): ', time.time()-start_time)