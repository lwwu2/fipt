import torch
import torch.nn as nn
import torch.nn.functional as NF

import tinycudann as tcnn

import math

import sys
sys.path.append('..')

from utils.ops import *


def diffuse_sampler(sample2,normal):
    """ sampling diffuse lobe: wi ~ NoV/math.pi 
    Args:
        sample2: Bx2 uniform samples
        normal: Bx3 normal
    Return:
        wi: Bx3 sampled direction in world space
    """
    theta = torch.asin(sample2[...,0].sqrt())
    phi = math.pi*2*sample2[...,1]
    wi = angle2xyz(theta,phi)
    
    Nmat = get_normal_space(normal)
    wi = (wi[:,None]@Nmat.permute(0,2,1)).squeeze(1)    
    return wi

def specular_sampler(sample2,roughness,wo,normal):
    """ sampling ggx lobe: h ~ D/(VoH*4)*NoH
    Args:
        sample2: Bx3 uniform samples
        roughness: Bx1 roughness
        wo: Bx3 viewing direction
        normal: Bx3 normal
    Return:
        wi: Bx3 sampled direction in world space
    """
    alpha = (roughness*roughness).squeeze(-1).data
    
    # sample half vector
    theta = (1-sample2[...,0])/(sample2[...,0]*(alpha*alpha-1)+1)
    theta = torch.acos(theta.sqrt())
    phi = 2*math.pi*sample2[...,1]
    wh = angle2xyz(theta,phi)

    # half vector to wi
    Nmat = get_normal_space(normal)
    wh = (wh[:,None]@Nmat.permute(0,2,1)).squeeze(1)
    wi = 2*(wo*wh).sum(-1,keepdim=True)*wh-wo
    wi = NF.normalize(wi,dim=-1)
    return wi



class BaseBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self,):
        super(BaseBRDF,self).__init__()
        return
    
    def forward(self,):
        pass
    
    def eval_diffuse(self,wi,normal):
        """ evaluate diffuse shading 
            and pdf
        """
        pdf = (normal*wi).sum(-1,keepdim=True).relu()/math.pi
        brdf = pdf.expand(len(wi),3) 
        return brdf,pdf
    
    def sample_diffuse(self,sample2,normal):
        """ sample diffuse shading
            and get sampled weight
        """
        # get wi
        wi = diffuse_sampler(sample2,normal)
        
        # get brdf/pdf, pdf
        brdf_weight = torch.ones(normal.shape,device=normal.device)
        pdf = (normal*wi).sum(-1,keepdim=True).relu()/math.pi
        return wi,pdf,brdf_weight
    
    def eval_specular(self,wi,wo,normal,roughness):
        """" evaluate specular shadings
            and pdf
        """
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        D = D_GGX(NoH,roughness)
        pdf = D.data/(4*VoH.clamp_min(1e-4))*NoH

        G = G_Smith(NoV,NoL,roughness)
        F0,F1 = fresnelSchlick_sep(VoH)
        
        # two term corresponds to two fresnel components
        brdf_spec0 = D*G*F0/4.0*NoL
        brdf_spec1 = D*G*F1/4.0*NoL

        return brdf_spec0,brdf_spec1,pdf 

    def sample_specular(self,sample2,wo,normal,roughness):
        """ evaluate specular shadings
            and get sampled weight
        """
        # get wi
        wi = specular_sampler(sample2,roughness,wo,normal)
        
        # get brdf/pdf, pdf
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()
        
        D = D_GGX(NoH,roughness)
        pdf = D.data/(4*VoH.clamp_min(1e-4))*NoH

        G = G_Smith(NoV,NoL,roughness)
        F0,F1 = fresnelSchlick_sep(VoH)
        
        fac = G*VoH*NoL/NoH.clamp_min(1e-4)
        
        brdf_weight0 = F0*fac
        brdf_weight1 = F1*fac
        return wi,pdf,brdf_weight0,brdf_weight1
    
    def eval_brdf(self,wi,wo,normal,mat):
        """ evaluate BRDF and pdf
        Args:
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: surface BRDF dict
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        albedo,roughness,metallic = mat['albedo'],mat['roughness'],mat['metallic']

        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()


        # get pdf
        D = D_GGX(NoH,roughness)
        pdf_spec = D.data/(4*VoH.clamp_min(1e-4))*NoH
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        # get brdf
        kd = albedo*(1-metallic)
        ks = 0.04*(1-metallic) + albedo*metallic

        G = G_Smith(NoV,NoL,roughness)
        F = fresnelSchlick(VoH,ks)
        brdf_diff = kd/math.pi*NoL
        brdf_spec = D*G*F/4.0*NoL

        brdf = brdf_diff + brdf_spec

        return brdf,pdf 
    
    def sample_brdf(self,sample1,sample2,wo,normal,mat):
        """ importance sampling brdf and get brdf/pdf
        Args:
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: material dict
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
            brdf_weight: Bx3 brdf/pdf
        """
        B = sample1.shape[0]
        device = sample1.device

        pdf = torch.zeros(B,device=device)
        brdf = torch.zeros(B,3,device=device)
        wi = torch.zeros(B,3,device=device)


        mask = (sample1 > 0.5)
        # sample diffuse
        wi[mask] = diffuse_sampler(sample2[mask],normal[mask])
        mask = ~mask
        # sample specular
        wi[mask] = specular_sampler(sample2[mask],mat['roughness'][mask],wo[mask],normal[mask])

        # get brdf,pdf
        brdf,pdf = self.eval_brdf(wi,wo,normal,mat)

        brdf_weight = torch.where(pdf>0,brdf/pdf,0)
        brdf_weight[brdf_weight.isnan()] = 0
        return wi,pdf,brdf_weight
    
    
class NGPBRDF(BaseBRDF):
    """ Hash Grid based brdf paramterization """
    def __init__(self,voxel_min,voxel_max):
        """ 
        voxel_min,voxel_max: scene bounding box
        """
        super(NGPBRDF,self).__init__()
        self.voxel_min = voxel_min
        self.voxel_max = voxel_max
        hash_encoding={
                "otype": "HashGrid",
                "n_levels": 32,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3
         }
        
        hash_network={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2
        }

        self.mlp =  tcnn.NetworkWithInputEncoding(
            3, 5, hash_encoding, hash_network)

        
    def forward(self,position):
        """ query brdf parameters at given location
        Args:
            position: Bx3 queried location
        Return:
            Bx3 base color
            Bx1 roughness in [0.02,1]
            Bx1 metallic
        """
        # map to [0,1]
        position = (position-self.voxel_min)/(self.voxel_max-self.voxel_min)
        
        mat = self.mlp(position*2-1).sigmoid()
        return {
            'albedo': mat[...,:3].float(),
            'roughness': mat[...,3:4].float()*0.98+0.02, # avoid nan
            'metallic': mat[...,4:5].float()
        }



class TextureBRDF(BaseBRDF):
    """ Textured mesh based brdf parameterization (unsued) """
    def __init__(self,res):
        super(TextureBRDF,self).__init__()
        self.res = res
        self.register_parameter('textures',nn.Parameter(torch.randn(self.res,self.res,3+2)*1e-2))
        
    def forward(self,uv):
        uv = (uv*(self.res-1)).clamp(0,self.res-1)
        uv0 = uv.floor().long()
        uv1 = uv.ceil().long()
        uv_ = uv-uv0
        
        u0,v0 = uv0.T
        u1,v1 = uv1.T
 
        
        t00,t01 = self.textures[v0,u0].sigmoid(),self.textures[v0,u1].sigmoid()
        t10,t11 = self.textures[v1,u0].sigmoid(),self.textures[v1,u1].sigmoid()
        
        u_,v_ = uv_.T
        u_,v_ = u_.unsqueeze(-1),v_.unsqueeze(-1)
        
        t = t00*(1-u_)*(1-v_) + t01*u_*(1-v_)\
          + t10*(1-u_)*v_ + t11*u_*v_
        
        return {
            'albedo': t[...,:3],
            'roughness': t[...,3:4]*0.98+0.02, # avoid nan
            'metallic': t[...,4:5]
        }  