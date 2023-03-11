
import torch

import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import trimesh
import numpy as np

import sys
sys.path.append('..')
from utils.dataset import RealDataset, SyntheticDataset
from utils.path_tracing import ray_intersect

from argparse import ArgumentParser
from tqdm import tqdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene',type=str,required=True,help='scene path')
    parser.add_argument('--dataset',type=str,required=True,help='scene type')


    args = parser.parse_args()
    device = torch.device(0)

    if args.dataset == 'real':
        dataset_fn = RealDataset
    elif args.dataset == 'synthetic':
        dataset_fn = SyntheticDataset

    # load dataset
    dataset = dataset_fn(args.scene,split='train',pixel=False,ray_diff=False)
    img_hw = dataset.img_hw

    # load scene geometry
    scene = mitsuba.load_dict({
                'type': 'scene',
                'shape_id':{
                    'type': 'obj',
                    'filename': os.path.join(args.scene, 'scene.obj')
                }
    })
    mesh = trimesh.load_mesh(os.path.join(args.scene,'scene.obj'))
    vertices = torch.from_numpy(np.array(mesh.vertices)).float()
    faces = torch.from_numpy(np.array(mesh.faces))

    # segmentation root folder
    if args.dataset == 'synthetic':
        args.scene = os.path.join(args.scene,'train')


    # find max segmentation indices
    seg_num = 0
    for i in tqdm(range(len(dataset))):
        segmentation = cv2.imread(os.path.join(args.scene,'segmentation/{:03d}.exr'.format(i)),-1)
        seg_num = max(int(segmentation.max()),seg_num)

    # build histogram of segmentation labels for each triangle
    labels = torch.zeros(len(faces)*seg_num,device=device,dtype=torch.long)
    for i in tqdm(range(len(dataset))):
        segmentation = cv2.imread(os.path.join(args.scene,'segmentation/{:03d}.exr'.format(i)),-1)
        segmentation = torch.from_numpy(segmentation[...,0]).long().reshape(-1)
        batch = dataset[i]
        rays = batch['rays'].to(device)
        xs,ds = rays[...,:3],rays[...,3:6]


        positions,normals,_,triangle_idx,valid = ray_intersect(scene,xs,ds)
        # flattened indices for histogram entries
        inds = triangle_idx[valid]*int(seg_num) + segmentation.to(device)[valid]
        labels.scatter_add_(0,inds,torch.ones_like(inds))

    # assign triangle label by maximum occurance
    new_label_num,new_label = labels.reshape(-1,seg_num)[:,1:].max(-1)
    new_label += 1
    # if the label occured 0 times, assign 0 label
    new_label[new_label_num==0] = 0


    # write fused segmentation
    OUTPUT_PATH = os.path.join(args.scene,'segmentation-new')
    os.makedirs(OUTPUT_PATH,exist_ok=True)
    im_id =0
    for batch in tqdm(dataset):
        rays = batch['rays'].to(device)
        xs = rays[...,:3]
        ds = rays[...,3:6]
        _,_,_,triangle_idx,valid = ray_intersect(scene,xs,ds)
        segmentation = new_label[triangle_idx]
        segmentation[~valid] = 0
        segmentation = segmentation.cpu().reshape(*img_hw,1).expand(*img_hw,3)
        cv2.imwrite(os.path.join(OUTPUT_PATH,'{:03d}.exr'.format(im_id)),segmentation.float().numpy())
        im_id += 1

    # move file folder
    os.system('mv {}/segmentation {}/segmentation-old'.format(args.scene,args.scene))
    os.system('mv {}/segmentation-new {}/segmentation'.format(args.scene,args.scene))