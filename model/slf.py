import torch
import torch.nn as nn
import torch.nn.functional as NF

""" surface light field model """
        
        
class VoxelSLF(nn.Module):
    """ voxel grid based surface light field """
    def __init__(self,mask,voxel_min,voxel_max):
        """
        mask: NxNxN voxel occupancy mask
        voxel_min,voxel_max: voxel bounding box
        """
        super(VoxelSLF,self).__init__()
        H = mask.shape[0]
        self.H = H
        self.voxel_min = voxel_min
        self.voxel_max = voxel_max
        
        # find coordinates for occupied voxels
        kk,jj,ii = torch.where(mask)
        inds = -torch.ones(H,H,H,dtype=torch.long)
        inds[kk,jj,ii] = torch.arange(len(ii))


        self.register_buffer('inds',inds)
        self.register_buffer(
            'radiance', torch.randn(len(ii),3)*1e-1)
        self.register_buffer(
            'count', torch.zeros(len(ii),dtype=torch.long)) # number of entries, used for mean pooling

    def spatial_idx(self,x):
        """ get voxel entry index for input location
        Args:
            x: Bx3 3D position
        Return:
            B indices
        """
        # map to voxel grid coordinates
        x_ = (x-self.voxel_min)/(self.voxel_max-self.voxel_min)
        x_ = (x_*self.H).long().clamp(0,self.H-1)

        # find entry indices
        idx = self.inds[x_[...,2],x_[...,1],x_[...,0]]
        return idx
        
    def scatter_add(self,x,radiance):
        """ scatter add radiance to voxel grid
        """
        idx = self.spatial_idx(x)
        self.radiance.scatter_add_(0,idx[...,None].expand_as(radiance),radiance)
        self.count.scatter_add_(0,idx,torch.ones_like(idx))
    
    def forward(self,x):
        """ query surface light field """
        idx = self.spatial_idx(x)
        radiance = self.radiance[idx]
        radiance[idx==-1] = 0 # if hit empty space, return zero radiance
        return {
            'rgb': radiance
        }


class TextureSLF(nn.Module):
    """ textured mesh based surface light field (unused) """
    def __init__(self,res,texture=None,Co=3):
        super(ExplicitSLF,self).__init__()
        self.res = res
        self.Co = Co
        if texture is None:
            texture = torch.randn(Co,res,res)*0.1 + 0.5
        self.register_parameter('feature',nn.Parameter(texture))
    
    def texture(self,uv):
        """ uv: Bx2"""
        uv = uv*2-1
        B,_ = uv.shape
        feat = NF.grid_sample(self.feature[None],uv.reshape(1,B,1,2),
                       mode='bilinear',align_corners=True).reshape(self.Co,B).T
        return feat
    
    def forward(self,uv,wo):
        feat = self.texture(uv)
        rgb = feat[...,:3]
        return {
            'rgb': rgb
        }