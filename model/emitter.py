import torch
import torch.nn as nn
import torch.nn.functional as NF
import numpy as np
import math

from .slf import VoxelSLF


class AreaEmitter(nn.Module):
    """ triangle mesh emitters """
    def __init__(self,emitter_path):
        """ emitter_path file 
        is_emitter: B indicator of whether a triangle is emitter
        emitter_vertices: Kx3x3 triangle vertices of emitters
        emitter_area: K surface areas of emitters
        emitter_radiance: Bx3x3 emitter radiance
        """
        super(AreaEmitter,self).__init__()
        
        weight = torch.load(emitter_path,map_location='cpu')
        
        is_emitter = weight['is_emitter']
        emitter_vertices = weight['emitter_vertices']
        emitter_area = weight['emitter_area']
        emitter_radiance = weight['emitter_radiance']

        self.register_buffer('is_emitter',is_emitter)
        self.register_buffer('emitter_vertices',emitter_vertices)
        self.register_buffer('emitter_area',emitter_area)
        self.register_buffer('radiance',emitter_radiance)
        
        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((len(is_emitter),),-1,device=is_emitter.device,dtype=torch.long)
        emitter_idx[is_emitter] = torch.arange(is_emitter.sum(),device=is_emitter.device)
        self.register_buffer('emitter_idx',emitter_idx)
        
        # emitter idx to triangle idx
        triangle_idx = torch.arange(len(is_emitter))[is_emitter]
        self.register_buffer('triangle_idx',triangle_idx)
        
        # sample emitters uniformly
        emitter_pdf = NF.normalize(torch.ones_like(emitter_area),dim=-1,p=1)
        emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        self.register_buffer('emitter_pdf',emitter_pdf)
        self.register_buffer('emitter_cdf',emitter_cdf)
    
    def forward(self,triangle_idx):
        """ get emitter radiance
        triangle_idx: B triangle indices
        """
        vis = triangle_idx != -1 # whether a valid triangle

        is_area = self.is_emitter[triangle_idx]&vis
        Le = torch.zeros(position.shape[0],3,device=position.device)
        if is_area.any():
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            Le[is_area] = self.radiance[e_idx]
        
        # assume zero background lighting
        Le = Le*vis[...,None]
        return Le
    
    def eval_emitter(self, position,light_dir,triangle_idx,*args):
        """ evaluate surface emission and pdf
        Args:
            position: Bx3 intersection location
            light_dir: Bx3 emission direction
            triangle_idx: B intersected triangle id
        Return:
            Le: Bx3 radiance
            emit_pdf: Bx1 emitter pdf
            valid_next: B valid surface
        """
        # whether valid intersection
        vis = triangle_idx != -1

        # get area light
        is_area = self.is_emitter[triangle_idx]&vis

        Le = torch.zeros(position.shape[0],3,device=position.device)
        emit_pdf = torch.zeros(position.shape[0],device=position.device)
        if is_area.any():
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            emit_pdf[is_area] = self.emitter_pdf[e_idx]/self.emitter_area[e_idx].clamp_min(1e-12)
            Le[is_area] = self.radiance[e_idx]

        # assume zero background lighting
        Le = Le*vis[...,None]

        # next: not area light or background
        valid_next = (~is_area)&vis
        return Le,emit_pdf.unsqueeze(-1),valid_next
    
    def sample_emitter(self,sample1,sample2,position):
        """ importance sampling emitters
        Args:
            sample1: B uniform samples
            sample2: Bx2 uniform samples
            position: Bx3 surfae location
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1 the sampling pdf (in area space)
            triangle_idx: B the sampled triangle id
        """
        # pick an emitter
        emitter_idx = torch.searchsorted(self.emitter_cdf,sample1.clamp_min(1e-12))
        pdf0 = self.emitter_pdf[emitter_idx]

        # unifromly sample points on triangles
        xi1 = sample2[...,0].sqrt()
        u = (1-xi1).unsqueeze(-1)
        v = (xi1*sample2[...,1]).unsqueeze(-1)
        w = 1-u-v

        # emitter area
        A1 = self.emitter_area[emitter_idx]
        # sampled location on triangle
        p1 = self.emitter_vertices[emitter_idx]
        p1 = p1[:,0]*u + p1[:,1]*v + p1[:,2]*w
        wi = NF.normalize(p1-position,dim=-1)
        triangle_idx = self.triangle_idx[emitter_idx]
        
        # pdf in area space
        pdf = pdf0/A1.clamp_min(1e-12)
        return wi,pdf.unsqueeze(-1),triangle_idx
    

class SLFEmitter(nn.Module):
    """ triangle emitters with diffuse radiance cache """
    def __init__(self,emitter_path,slf_path):
        """ 
        emitter_path: emitter parameter file
        slf_path: surface light field paramter file
        """
        super(SLFEmitter,self).__init__()
        
        # load surface light field
        state_dict = torch.load(slf_path,map_location='cpu') 
        self.slf = VoxelSLF(state_dict['mask'],
                            state_dict['voxel_min'],state_dict['voxel_max'])
        self.slf.load_state_dict(state_dict['weight'])
        
        # load emitters
        weight = torch.load(emitter_path,map_location='cpu')
        is_emitter = weight['is_emitter']
        emitter_vertices = weight['emitter_vertices']
        emitter_area = weight['emitter_area']
        emitter_radiance = weight['emitter_radiance']
        self.register_buffer('is_emitter',is_emitter)
        self.register_buffer('emitter_vertices',emitter_vertices)
        self.register_buffer('emitter_area',emitter_area)
        self.register_buffer('radiance',emitter_radiance)
        
        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((len(is_emitter),),-1,device=is_emitter.device,dtype=torch.long)
        emitter_idx[is_emitter] = torch.arange(is_emitter.sum(),device=is_emitter.device)
        self.register_buffer('emitter_idx',emitter_idx)
        
        # emitter idx to triangle idx
        triangle_idx = torch.arange(len(is_emitter))[is_emitter]
        self.register_buffer('triangle_idx',triangle_idx)
        
        # randomly select a emitter
        emitter_pdf = NF.normalize(torch.ones_like(emitter_area),dim=-1,p=1)
        emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        self.register_buffer('emitter_pdf',emitter_pdf)
        self.register_buffer('emitter_cdf',emitter_cdf)
    
    def forward(self,position):
        """ surface light field from queried location """
        Le = self.slf(position)['rgb']
        return Le
    
    def eval_emitter(self, position,light_dir,triangle_idx,roughness=None):
        """ evaluate surface emission and pdf return radiance cache if diffuse
        Args:
            position: Bx3 intersection location
            light_dir: Bx3 emission direction
            triangle_idx: B intersected triangle id
            roughness: Bx1 surface roughness if not None
        Return:
            Le: Bx3 radiance
            emit_pdf: Bx1 emitter pdf
            valid_next: B valid surface
        """
        # whether valid intersection
        vis = triangle_idx != -1
        
        Le = torch.zeros(position.shape[0],3,device=position.device)
        emit_pdf = torch.zeros(position.shape[0],device=position.device)
        
        # get area light
        is_area = self.is_emitter[triangle_idx]&vis
        if is_area.any():
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            emit_pdf[is_area] = self.emitter_pdf[e_idx]/self.emitter_area[e_idx].clamp_min(1e-12)
            Le[is_area] = self.radiance[e_idx]
        
        # assume zero background lighting
        Le = Le*vis[...,None]
        valid_next = (~is_area)&vis
        
        # check diffuse radiance cache
        if roughness is not None:
            # query the radiance cache and terminate for diffuse and non emissive surface 
            is_diffuse = (~is_area) & vis & (roughness.squeeze(-1)>0.6)
            if is_diffuse.any():
                diffuse_slf = self.slf(position[is_diffuse])['rgb']
                Le[is_diffuse] = diffuse_slf
                is_diffuse[is_diffuse.clone()] = diffuse_slf.sum(-1)>0 # diffuse radiance need to > 0
                valid_next &= (~is_diffuse) # terminate path 

        return Le,emit_pdf.unsqueeze(-1),valid_next
    

    def sample_emitter(self,sample1,sample2,position):
        """ importance sampling emitters
        Args:
            sample1: B uniform samples
            sample2: Bx2 uniform samples
            position: Bx3 surfae location
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1 the sampling pdf (in area space)
            triangle_idx: B the sampled triangle id
        """
        # pick an emitter
        emitter_idx = torch.searchsorted(self.emitter_cdf,sample1.clamp_min(1e-12))
        pdf0 = self.emitter_pdf[emitter_idx]

        # unifromly sample points on triangles
        xi1 = sample2[...,0].sqrt()
        u = (1-xi1).unsqueeze(-1)
        v = (xi1*sample2[...,1]).unsqueeze(-1)
        w = 1-u-v

        # emitter area
        A1 = self.emitter_area[emitter_idx]
        # sampled location on triangle
        p1 = self.emitter_vertices[emitter_idx]
        p1 = p1[:,0]*u + p1[:,1]*v + p1[:,2]*w
        wi = NF.normalize(p1-position,dim=-1)
        triangle_idx = self.triangle_idx[emitter_idx]
        
        # pdf in area space
        pdf = pdf0/A1.clamp_min(1e-12)
        return wi,pdf.unsqueeze(-1),triangle_idx

