import torch
import torch.nn.functional as NF
import math


def get_normal_space(normal):
    """ get matrix transform shading space to normal spanned space
    Args:
        normal: Bx3
    Return:
        Bx3x3 transformation matrix
    """
    v1 = torch.zeros_like(normal)
    tangent = v1.clone()
    v1[...,0] = 1.0
    tangent[...,1] = 1.0
    
    mask = (v1*normal).sum(-1).abs() <= 1e-1
    tangent[mask] = NF.normalize(torch.cross(v1[mask],normal[mask],dim=-1),dim=-1)
    mask = ~mask
    tangent[mask] = NF.normalize(torch.cross(tangent[mask],normal[mask],dim=-1),dim=-1)
    
    bitangent = torch.cross(normal,tangent,dim=-1)
    return torch.stack([tangent,bitangent,normal],dim=-1)

def angle2xyz(theta,phi):
    """ spherical coordinates to euclidean 
    Args:
        theta,phi: B
    Return:
        Bx3 euclidean coordinates
    """
    sin_theta = torch.sin(theta)
    x = sin_theta*torch.cos(phi)
    y = sin_theta*torch.sin(phi)
    z = torch.cos(theta)
    ret = torch.stack([x,y,z],dim=-1)
    return NF.normalize(ret,dim=-1)

def G1_GGX_Schlick(NoV, eta):
    """ G term of schlick GGX
    eta: roughness
    """
    r = eta
    k = (r+1)
    k = k*k/8
    denom = NoV*(1-k)+k
    return 1 /denom

def G_Smith(NoV,NoL,eta):
    """ Smith shadow masking divided by (NoV*NoL)
    eta: roughness 
    """
    g1_l = G1_GGX_Schlick(NoL,eta)
    g1_v = G1_GGX_Schlick(NoV,eta)
    return g1_l*g1_v

def fresnelSchlick(VoH,F0):
    """ schlick fresnel """
    x = (1-VoH).pow(5)
    return F0 + (1-F0)*x

def fresnelSchlick_sep(VoH):
    """ two terms of schlick fresnel """
    x = (1-VoH).pow(5)
    return (1-x),x

def D_GGX(cos_h,eta):
    """GGX normal distribution
    eta: roughness
    """
    alpha = eta*eta
    alpha2 = alpha*alpha
    denom = (cos_h*cos_h*(alpha2-1.0)+1.0)
    denom = math.pi * denom*denom
    return alpha2/denom


def double_sided(V,N):
    """ double sided normal 
    Args:
        V: Bx3 viewing direction
        N: Bx3 normal direction
    Return:
        Bx3 flipped normal towards camera direction
    """
    NoV = (N*V).sum(-1)
    flipped = NoV<0
    N[flipped] = -N[flipped]
    return N

    
def lerp_specular(specular,roughness):
    """ interpolate specular shadings by roughness
    Args:
        specular: Bx6x3 specular shadings
        roughness: Bx1 roughness in [0.02,1.0]
    Return:
        Bx3 interpolated specular shading
    """
    # remap roughness from to [0,1]
    r_min,r_max = 0.02,1.0 
    r_num = specular.shape[-2]
    r = (roughness-r_min)/(r_max-r_min)*(r_num-1)
    
    
    r1 = r.ceil().long()
    r0 = r.floor().long()
    r_ = (r-r0)
    s0 = torch.gather(specular,1,r0[...,None].expand(r0.shape[0],1,3))[:,0]
    s1 = torch.gather(specular,1,r1[...,None].expand(r1.shape[0],1,3))[:,0]
    s = s0*(1-r_) + s1*r_
    return s