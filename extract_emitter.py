import torch
import torch.nn.functional as NF
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

from utils.dataset import RealDataset,SyntheticDataset
from utils.path_tracing import ray_intersect
from model.mlps import ImplicitMLP

import math
import os
import numpy as np
import trimesh
import time
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene', type=str, required=True, help= 'dataset folder')
    parser.add_argument('--output', type=str, required=True, help='output path')
    parser.add_argument('--ckpt', type=str, required=True,help='checkpoint path')
    parser.add_argument('--dataset',type=str,required=True, help='dataset type')
    parser.add_argument('--spp', type=int,default=100, help='number of samples for each triangle emitter')
    parser.add_argument('--threshold',type=float,default=0.01,help='threshold for emitter')

    args = parser.parse_args()

    device = torch.device(0)

    SCENE = args.scene
    CKPT = args.ckpt
    OUTPUT =args.output

    # load geometry
    scene = mitsuba.load_dict({
        'type': 'scene',
        'shape_id':{
            'type': 'obj',
            'filename':os.path.join(SCENE,'scene.obj')
        }
    })

    if args.dataset == 'synthetic':
        dataset_fn = SyntheticDataset
    elif args.dataset == 'real':
        dataset_fn = RealDataset
    dataset = dataset_fn(SCENE,split='train',pixel=False)
    img_hw = dataset.img_hw


    # get mesh vertices and triangles
    mesh = trimesh.load_mesh(os.path.join(SCENE,'scene.obj'))
    vertices = torch.from_numpy(np.array(mesh.vertices)).float()
    faces = torch.from_numpy(np.array(mesh.faces))


    # load emission mask
    state_dict = torch.load(CKPT,map_location='cpu')['state_dict']
    
    weight = {}
    for k,v in state_dict.items():
        if 'emission_mask' in k:
            weight[k.replace('emission_mask.','')] = v
    
    emission_net = ImplicitMLP(6,128,[3],1,10)
    emission_net.load_state_dict(weight)
    emission_net.to(device)
    for p in emission_net.parameters():
        p.requires_grad = False


    # sample emission mask on triangles
    spp = args.spp
    
    # batched processing emission mask
    B = len(faces)
    emit = torch.zeros(B)
    batch_size = 10240//spp
    for b in tqdm(range(math.ceil(B*1.0/batch_size))):
        b0 = b*batch_size
        b1 = min(B,b0+batch_size)

        # sample points on triangle
        v0,v1 = torch.rand(2,b1-b0,spp,device=device)
        v0,v1 = 1-v0.sqrt(),v0.sqrt()*v1
        v2 = 1-v0-v1
        p_emit = torch.einsum(
            'bkc,bsk->bsc',
            vertices[faces[b0:b1]].to(device),
            torch.stack([v0,v1,v2],-1)
        ).reshape(-1,3)

        # get emission mask
        emit_ = emission_net(p_emit).relu().squeeze(-1)
        emit_ = 1-torch.exp(-emit_)
        emit_ = emit_.reshape(-1,spp).mean(-1)
        emit[b0:b1] = emit_.cpu()
    
    # find all the emitters
    is_emitter = emit>args.threshold
    emitter_vertices = vertices[faces[is_emitter]]
    emitter_area = torch.cross(emitter_vertices[:,1]-emitter_vertices[:,0],
                emitter_vertices[:,2]-emitter_vertices[:,0],-1)
    emitter_normal = NF.normalize(emitter_area,dim=-1)
    emitter_area = emitter_area.norm(dim=-1)/2.0


    # assume constant emission
    emitter_idx = torch.full((len(is_emitter),),-1,device=is_emitter.device,dtype=torch.long)
    emitter_idx[is_emitter] = torch.arange(is_emitter.sum(),device=is_emitter.device)
    emitter_radiance = torch.zeros(is_emitter.sum(),3)
    emitter_radiance_count = torch.zeros(is_emitter.sum())


    # median pool the pixel radiance to estimat emission
    emitter_radiance = []
    emitter_radiance_idx = []
    for batch in tqdm(dataset):
        rays = batch['rays']
        rays_x,rays_d = rays[...,:3].to(device),rays[...,3:6].to(device)
        positions,normals,uvs,triangle_idxs,valid = ray_intersect(scene,rays_x,rays_d)
        
        is_area = is_emitter[triangle_idxs[valid].cpu()]
        if not is_area.any():
            continue
        is_area_idx = emitter_idx[triangle_idxs[valid].cpu()[is_area]]
        radiance = batch['rgbs'][valid.cpu()][is_area]
        
        emitter_radiance.append(radiance)
        emitter_radiance_idx.append(is_area_idx)

    emitter_radiance = torch.cat(emitter_radiance,0)
    emitter_radiance_idx = torch.cat(emitter_radiance_idx,0)

    # median pool
    emitter_radiance_out = torch.zeros(len(emitter_area),3)
    for i in tqdm(range(len(emitter_area))):
        mask = (emitter_radiance_idx == i)
        if mask.any():
            emitter_radiance_out[i] = emitter_radiance[mask].median(0)[0]

    # drop zero radiance emitter
    is_emitter_new = emitter_radiance_out.mean(-1) > 0
    is_emitter[is_emitter.clone()] = is_emitter_new
    emitter_vertices = emitter_vertices[is_emitter_new]
    emitter_area = emitter_area[is_emitter_new]
    emitter_normal = emitter_normal[is_emitter_new]
    emitter_radiance = emitter_radiance_out[is_emitter_new]


    torch.save({
        'is_emitter': is_emitter,
        'emitter_vertices': emitter_vertices,
        'emitter_area': emitter_area,
        'emitter_normal': emitter_normal,
        'emitter_radiance': emitter_radiance
    }, os.path.join(OUTPUT,'emitter.pth'))