
import torch
import torch.nn.functional as NF
import torch.optim as optim
from torch.utils.data import DataLoader

import torch_scatter

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

import math

import os
from pathlib import Path
from argparse import Namespace, ArgumentParser


from configs.config import default_options
from utils.dataset import InvRealDataset,RealDataset,InvSyntheticDataset,SyntheticDataset
from utils.ops import *
from utils.path_tracing import ray_intersect
from model.mlps import ImplicitMLP
from model.brdf import NGPBRDF

class ModelTrainer(pl.LightningModule):
    """ BRDF-emission mask training code """
    def __init__(self, hparams: Namespace, *args, **kwargs):
        super(ModelTrainer, self).__init__()
        self.save_hyperparameters(hparams)
        
        # load scene geometry
        self.scene = mitsuba.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': 'obj',
                'filename': os.path.join(hparams.dataset[1],'scene.obj')
            }
        })

        # initiallize BRDF
        mask = torch.load(hparams.voxel_path,map_location='cpu')
        self.material = NGPBRDF(mask['voxel_min'],mask['voxel_max'])
       
        # intiialize emission mask
        self.emission_mask = ImplicitMLP(6,128,[3],1,10)
        
        
    def __repr__(self):
        return repr(self.hparams)

    def configure_optimizers(self):
        if(self.hparams.optimizer == 'SGD'):
            opt = optim.SGD
        if(self.hparams.optimizer == 'Adam'):
            opt = optim.Adam
        
        optimizer = opt(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)    
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.hparams.milestones,gamma=self.hparams.scheduler_rate)
        return [optimizer], [scheduler]
    
    def train_dataloader(self,):
        dataset_name,dataset_path,cache_path = self.hparams.dataset
        
        if dataset_name == 'synthetic':
            dataset = InvSyntheticDataset(dataset_path,cache_path,pixel=True,split='train',
                                       batch_size=self.hparams.batch_size,has_part=self.hparams.has_part)
        elif dataset_name == 'real':
            dataset = InvRealDataset(dataset_path,cache_path,pixel=True,split='train',
                                       batch_size=self.hparams.batch_size,has_part=self.hparams.has_part)
       
        return DataLoader(dataset, batch_size=None)
       
    def on_train_epoch_start(self,):
        """ resample training batch """
        self.train_dataloader().dataset.resample()
    
    def val_dataloader(self,):
        dataset_name,dataset_path,cache_path = self.hparams.dataset
        self.dataset_name = dataset_name

        if dataset_name == 'synthetic':
            dataset = SyntheticDataset(dataset_path,pixel=False,split='val')
        elif dataset_name == 'real':
            dataset = RealDataset(dataset_path,pixel=False,split='val')
        
        self.img_hw = dataset.img_hw
        return DataLoader(dataset, shuffle=False,batch_size=None)

    def forward(self, points, view):
        return

    def gamma(self,x):
        mask = x <= 0.0031308
        ret = torch.empty_like(x)
        ret[mask] = 12.92*x[mask]
        mask = ~mask
        ret[mask] = 1.055*x[mask].pow(1/2.4) - 0.055
        return ret
    
    def training_step(self, batch, batch_idx):
        """ one training step """
        rays,rgbs_gt = batch['rays'], batch['rgbs']
        xs,ds = rays[...,:3],rays[...,3:6]
        ds = NF.normalize(ds,dim=-1)
        
        if self.dataset_name == 'synthetic': # only available for synthetic scene
            albedos_gt = batch['albedo']

        # fetch shadings
        diffuse = batch['diffuse']
        specular0 = batch['specular0']
        specular1 = batch['specular1']
        
        # fetch segmentation
        segmentation = batch['segmentation'].long()
        
        # find surface intersection
        positions,normals,_,_,valid = ray_intersect(self.scene,xs,ds)

        if not valid.any():
            return None
        
        # optimize only valid surface
        normals = normals[valid]
        rgbs_gt = rgbs_gt[valid]
        positions = positions[valid]
        diffuse=diffuse[valid]
        specular0 = specular0[valid]
        specular1 = specular1[valid]
        if self.dataset_name == 'synthetic':
            albedos_gt = albedos_gt[valid]
  
        segmentation = segmentation[valid]
        
        # get brdf
        mat = self.material(positions)
        albedo,metallic,roughness = mat['albedo'],mat['metallic'],mat['roughness']
        
        # diffuse and specular reflectance
        kd = albedo*(1-metallic)
        ks = 0.04*(1-metallic) + albedo*metallic
       
        # diffuse component and specular component
        Ld = kd*diffuse
        Ls = ks*lerp_specular(specular0,roughness)+lerp_specular(specular1,roughness)
        rgbs = Ld+Ls
        

        # get emission mask
        emission_mask = self.emission_mask(positions)
        alpha = (1-torch.exp(-emission_mask.relu()))    
        
        
        # mask out emissive regions
        rgbs = (1-alpha)*rgbs+rgbs_gt*alpha
        # tonemapped mse loss
        loss_c = NF.mse_loss(self.gamma(rgbs),self.gamma(rgbs_gt))
        
        # regualrize emission mask to be small
        loss_e = self.hparams.le * emission_mask.abs().mean()
        
        # diffuse regualrization
        loss_d = self.hparams.ld * ((roughness-1).abs().mean()+metallic.mean())

        # roughness-metallic propagation regularization
        if self.hparams.has_part:
            # with part segmentation 

            # find mean roughness-metallic for each segmentation id
            seg_idxs,inv_idxs = segmentation.unique(return_inverse=True)
            weight_seg = torch.zeros(len(seg_idxs),device=seg_idxs.device)
            mean_metallic = torch.zeros(len(seg_idxs),device=seg_idxs.device)
            mean_roughness = torch.zeros(len(seg_idxs),device=seg_idxs.device)
            
            weight_seg_ = Ls.data.mean(-1)+1e-4 # weight surface with high reflection more
            
            mean_metallic = torch_scatter.scatter(
                metallic.squeeze(-1)*weight_seg_,inv_idxs,0,mean_metallic,reduce='sum').unsqueeze(-1)
            mean_roughness = torch_scatter.scatter(
                roughness.squeeze(-1)*weight_seg_,inv_idxs,0,mean_roughness,reduce='sum').unsqueeze(-1)
            weight_seg = torch_scatter.scatter(weight_seg_,inv_idxs,0,weight_seg,reduce='sum').unsqueeze(-1)


            mean_metallic = mean_metallic/weight_seg
            mean_roughness = mean_roughness/weight_seg
    
            # propagation loss
            loss_seg = (metallic-mean_metallic[inv_idxs]).abs().mean()\
                     + (roughness-mean_roughness[inv_idxs]).abs().mean()
            loss_seg = self.hparams.lp*loss_seg
        else:
            # with semantic segmentation

            # normalize input position
            positions = (positions-self.material.voxel_min)/(self.material.voxel_max-self.material.voxel_min)*2-1

            # find mean amount all the pixels is expensive, only sample subset (1024) of them
            seg_idxs,inv_idxs,seg_counts = segmentation.unique(return_inverse=True,return_counts=True)
            ii,jj = [],[]
            for seg_idx,seg_count in zip(seg_idxs,seg_counts):
                sample_batch = 1024
                i = torch.where(segmentation==seg_idx)[0]
                if sample_batch > seg_count:
                    sample_batch = seg_count
                    j = torch.arange(seg_count,device=seg_idxs.device)[None].repeat_interleave(sample_batch,0).reshape(-1)
                else:
                    j = torch.randint(0,seg_count,(seg_count*sample_batch,),device=seg_idxs.device)
                j = i[j]
                i = i.repeat_interleave(sample_batch,0)
                ii.append(i)
                jj.append(j)
            ii = torch.cat(ii,0)
            jj = torch.cat(jj,0)


            # weight more of close pixels with similar albedo
            weight_seg_ = torch.exp(-(
                        (albedo.data[ii]-albedo.data[jj]).pow(2).sum(-1)
                        /self.hparams.sigma_albedo**2)/2.0)
            weight_seg_*= torch.exp(-((positions[ii]-positions[jj]).pow(2).sum(-1)
                        /self.hparams.sigma_pos**2)/2.0)


            weight_seg = torch.zeros(len(positions),device=positions.device)+1e-4
            roughness_mean = torch.zeros(len(roughness),device=roughness.device)
            metallic_mean = torch.zeros(len(metallic),device=metallic.device)
            
            # calculate mean for each pixel
            roughness_mean.scatter_add_(0,ii,roughness[jj].squeeze(-1)*weight_seg_)
            metallic_mean.scatter_add_(0,ii,metallic[jj].squeeze(-1)*weight_seg_)
            weight_seg.scatter_add_(0,ii,weight_seg_)
            roughness_mean = roughness_mean/weight_seg
            metallic_mean = metallic_mean/weight_seg


            loss_seg_ = (roughness_mean-roughness.squeeze(-1)).abs()+(metallic_mean-metallic.squeeze(-1)).abs()
            
            # propagation
            loss_seg = torch.zeros(len(seg_idxs),device=seg_idxs.device)
            loss_seg = torch_scatter.scatter(loss_seg_,inv_idxs,0,loss_seg,reduce='mean')
            loss_seg = self.hparams.ls*loss_seg.sum()

        # vsualize rendering brdf
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        loss = loss_c + loss_e + loss_d + loss_seg
        
        if self.dataset_name == 'synthetic':
            albedo_loss = NF.mse_loss(albedos_gt,kd.data)
            self.log('train/albedo', albedo_loss)
        self.log('train/loss', loss)
        self.log('train/psnr', psnr)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ visualize diffuse reflectance kd
        """
        rays,rgb_gt = batch['rays'], batch['rgbs']
        if self.dataset_name == 'synthetic':
            emission_mask_gt = batch['emission'].mean(-1,keepdim=True) == 0
        else:
            emission_mask_gt = torch.ones_like(rays[...,:1])
        rays_x = rays[:,:3]
        rays_d = NF.normalize(rays[:,3:6],dim=-1)

        positions,normals,_,_,valid = ray_intersect(self.scene,rays_x,rays_d)
        position = positions[valid]

        # batched rendering diffuse reflectance
        B = valid.sum()
        batch_size = 10240
        albedo_ = []
        for b in range(math.ceil(B*1.0/batch_size)):
            b0 = b*batch_size
            b1 = min(b0+batch_size,B)
            mat = self.material(position[b0:b1])
            albedo_.append(mat['albedo']*(1-mat['metallic']))
        albedo_ = torch.cat(albedo_)
        albedo = torch.zeros(len(valid),3,device=valid.device)
        albedo[valid] = albedo_
        
        if self.dataset_name == 'synthetic':
            albedo_gt = batch['albedo']
        else: # show rgb is no ground truth kd
            albedo_gt = rgb_gt.pow(1/2.2).clamp(0,1)

        # mask out emissive regions
        albedo = albedo*emission_mask_gt
        albedo_gt = albedo_gt * emission_mask_gt
        loss_c = NF.mse_loss(albedo_gt,albedo)
        
        loss = loss_c
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        
        
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)

        self.logger.experiment.add_image('val/gt_image', albedo_gt.reshape(*self.img_hw,3).permute(2, 0, 1).clamp(0,1), batch_idx)
        self.logger.experiment.add_image('val/inf_image', albedo.reshape(*self.img_hw,3).permute(2, 0, 1).clamp(0,1), batch_idx)
        return

            
def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser
        
if __name__ == '__main__':

    torch.manual_seed(9)
    torch.cuda.manual_seed(9)

    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--ft', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--device', type=int, required=False,default=None)

    parser.set_defaults(resume=False)
    args = parser.parse_args()
    args.gpus = [args.device]
    experiment_name = args.experiment_name

    # setup checkpoint loading
    checkpoint_path = Path(args.checkpoint_path) / experiment_name
    log_path = Path(args.log_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val/loss', save_top_k=1, save_last=True)
    logger = TensorBoardLogger(log_path, name=experiment_name)

    last_ckpt = checkpoint_path / 'last.ckpt' if args.resume else None
    if (last_ckpt is None) or (not (last_ckpt.exists())):
        last_ckpt = None
    else:
        last_ckpt = str(last_ckpt)
    
    # setup model trainer
    model = ModelTrainer(hparams)
    
    trainer = Trainer.from_argparse_args(
        args,
        resume_from_checkpoint=last_ckpt,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        flush_logs_every_n_steps=1,
        log_every_n_steps=1,
        max_epochs=args.max_epochs
    )

    trainer.fit(model)