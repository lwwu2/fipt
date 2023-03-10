import torch
import torch.nn.functional as NF

import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

import os
import sys
sys.path.append('..')
from model.brdf import NGPBRDF
from model.emitter import SLFEmitter

class FIPTBSDF(mitsuba.BSDF):
    def __init__(self, props):
        mitsuba.BSDF.__init__(self, props)
        # default device for mitsuba
        device = torch.device(0)

        # load BRDF and emission mask
        mask = torch.load(os.path.join(props['emitter_path'],'vslf.npz'),map_location='cpu')
        state_dict = torch.load(props['brdf_path'],map_location='cpu')['state_dict']
        weight = {}
        for k,v in state_dict.items():
            if 'material.' in k:
                weight[k.replace('material.','')]=v
        material_net = NGPBRDF(mask['voxel_min'],mask['voxel_max'])
        material_net.load_state_dict(weight)
        material_net.to(device)
        for p in material_net.parameters():
            p.requires_grad=False
        self.material_net = material_net
        self.is_emitter = torch.load(os.path.join(props['emitter_path'],'emitter.pth'))['is_emitter'].to(device)
        
        # specify flags
        reflection_flags   = mitsuba.BSDFFlags.SpatiallyVarying|mitsuba.BSDFFlags.DiffuseReflection|mitsuba.BSDFFlags.FrontSide | mitsuba.BSDFFlags.BackSide
        self.m_components  = [reflection_flags]
        self.m_flags = reflection_flags

    def sample(self, ctx, si, sample1, sample2, active):
        wi = si.to_world(si.wi).torch()
        normal = si.n.torch()
        position = si.p.torch()
        triangle_idx = mitsuba.Int(si.prim_index).torch().long()
        
        mat = self.material_net(position)
        wo,pdf,brdf_weight = self.material_net.sample_brdf(
            sample1.torch().reshape(-1),sample2.torch(),
            wi,normal,mat
        )
        brdf_weight[self.is_emitter[triangle_idx]] = 0
        
        pdf_mi = mitsuba.Float(pdf.squeeze(-1))
        wo_mi = mitsuba.Vector3f(wo[...,0],wo[...,1],wo[...,2])
        wo_mi = si.to_local(wo_mi)
        value_mi = mitsuba.Vector3f(brdf_weight[...,0],brdf_weight[...,1],brdf_weight[...,2])
        
        bs = mitsuba.BSDFSample3f()
        bs.pdf = pdf_mi
        bs.sampled_component = mitsuba.UInt32(0)
        bs.sampled_type = mitsuba.UInt32(+self.m_flags)
        bs.wo = wo_mi
        bs.eta = 1.0

        return (bs,value_mi)

    def eval(self, ctx, si, wo, active):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()
        triangle_idx = mitsuba.Int(si.prim_index).torch().long()
        
        normal = si.n.torch()
        position = si.p.torch()
        
        mat = self.material_net(position)
        
        brdf,_ = self.material_net.eval_brdf(wo,wi,normal,mat)
        brdf[self.is_emitter[triangle_idx]]=0
        brdf = mitsuba.Vector3f(brdf[...,0],brdf[...,1],brdf[...,2])
        
        return brdf


    def pdf(self, ctx, si, wo,active):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()
        
        normal = si.n.torch()
        position = si.p.torch()
        
        mat = self.material_net(position)
        _,pdf = self.material_net.eval_brdf(wo,wi,normal,mat)
        pdf = mitsuba.Float(pdf.squeeze(-1))
        return pdf


    def eval_pdf(self, ctx, si, wo, active=True):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()
        triangle_idx = mitsuba.Int(si.prim_index).torch().long()
        
        normal = si.n.torch()
        position = si.p.torch()
        
        mat = self.material_net(position)
        
        brdf,pdf = self.material_net.eval_brdf(wo,wi,normal,mat)
        brdf[self.is_emitter[triangle_idx]] = 0
        brdf = mitsuba.Vector3f(brdf[...,0],brdf[...,1],brdf[...,2])
        pdf = mitsuba.Float(pdf.squeeze(-1))
        
        return brdf,pdf
    def to_string(self,):
        return 'FIPTBSDF'


mitsuba.register_bsdf("fipt", lambda props: FIPTBSDF(props))