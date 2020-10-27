# -*- coding: utf-8 -*-

import os, pprint, tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu 
from haven import haven_img as hi
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import fcn8_vgg16
from src import utils as ut
from src import models
from . import lcfcn
import sys
import kornia
from kornia.augmentation import RandomAffine
from scipy.ndimage.filters import gaussian_filter
from kornia.geometry.transform import flips
from . import optimizers, metrics, networks

class SemSeg(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.exp_dict = exp_dict
        self.train_hashes = set()
        self.n_classes = self.exp_dict['model'].get('n_classes', 1)

        self.init_model()
        self.first_time = True
        self.epoch = 0

    def init_model(self):
        self.model_base = networks.get_network(self.exp_dict['model']['base'],
                                              n_classes=self.n_classes,
                                              exp_dict=self.exp_dict)
        self.cuda()
        self.opt = optimizers.get_optimizer(self.exp_dict['optimizer'], self.model_base, self.exp_dict)


    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict(),
                      'epoch':self.epoch}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        if 'opt' not in state_dict:
            return
        self.opt.load_state_dict(state_dict["opt"])
        self.epoch = state_dict['epoch']

    def train_on_loader(self, train_loader):
        
        self.train()
        self.epoch += 1
        n_batches = len(train_loader)

        pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
        train_monitor = TrainMonitor()
    
        for batch in train_loader:
            score_dict = self.train_on_batch(batch)
            train_monitor.add(score_dict)
            msg = ' '.join(["%s: %.3f" % (k, v) for k,v in train_monitor.get_avg_score().items()])
            pbar.set_description('Training - %s' % msg)
            pbar.update(1)
            
        pbar.close()

        if self.exp_dict.get('adjust_lr'):
            infnet.adjust_lr(self.opt,
                                self.epoch, decay_rate=0.1, decay_epoch=30)

        return train_monitor.get_avg_score()

    def compute_mask_loss(self, loss_name, images, logits, masks):
        if loss_name == 'cross_entropy':
            if self.n_classes == 1:
                loss = F.binary_cross_entropy_with_logits(logits, masks.float(), reduction='mean')
            else:
                probs = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(probs, masks, reduction='mean', ignore_index=255)

        elif loss_name == 'joint_cross_entropy':
            loss = ut.joint_loss(logits, masks.float())
        
        return loss 

    def compute_point_loss(self, loss_name, images, logits, points):
        if loss_name == 'toponet':
            if self.first_time:
                self.first_time = False
                self.vgg = nn.DataParallel(lanenet.VGG().cuda(1), list(range(1,4)))
                self.vgg.train()
            points = points[:,None]
            images_flip = flips.Hflip()(images)
            logits_flip = self.model_base(images_flip)
            loss = torch.mean(torch.abs(flips.Hflip()(logits_flip)-logits))
            
            logits_flip_vgg = self.vgg(F.sigmoid(logits_flip.cuda(1)))
            logits_vgg = self.vgg(F.sigmoid(logits.cuda(1))) 
            loss += self.exp_dict["model"]["loss_weight"] * torch.mean(torch.abs(flips.Hflip()(logits_flip_vgg)-logits_vgg)).cuda(0)

            ind = points!=255
            if ind.sum() != 0:
                loss += F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')

                points_flip = flips.Hflip()(points)
                ind = points_flip!=255
                loss += F.binary_cross_entropy_with_logits(logits_flip[ind], 
                                        points_flip[ind].float().cuda(), 
                                        reduction='mean')

        elif loss_name == 'multiscale_cons_point_loss':
            logits, features = logits
            points = points[:,None]
            ind = points!=255
            # self.vis_on_batch(batch, savedir_image='tmp.png')

            # POINT LOSS
            # loss = ut.joint_loss(logits, points[:,None].float().cuda(), ignore_index=255)
            # print(points[ind].sum())
            logits_flip, features_flip = self.model_base(flips.Hflip()(images), return_features=True)

            loss = torch.mean(torch.abs(flips.Hflip()(logits_flip)-logits))
            for f, f_flip in zip(features, features_flip):
                loss += torch.mean(torch.abs(flips.Hflip()(f_flip)-f)) * self.exp_dict["model"]["loss_weight"]
            if ind.sum() != 0:
                loss += F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')

                # logits_flip = self.model_base(flips.Hflip()(images))
                points_flip = flips.Hflip()(points)
                ind = points_flip!=255
                # if 1:
                #     pf = points_flip.clone()
                #     pf[pf==1] = 2
                #     pf[pf==0] = 1
                #     pf[pf==255] = 0
                #     lcfcn_loss.save_tmp('tmp.png', flips.Hflip()(images[[0]]), logits_flip[[0]], 3, pf[[0]])
                loss += F.binary_cross_entropy_with_logits(logits_flip[ind], 
                                        points_flip[ind].float().cuda(), 
                                        reduction='mean')

        elif loss_name == "affine_cons_point_loss":
            points = points[:,None]
            if np.random.randint(2) == 0:
                images_flip = flips.Hflip()(images)
                flipped = True
            else:
                images_flip = images
                flipped = False
            batch_size, C, height, width = logits.shape
            random_affine = RandomAffine(degrees=2, translate=None, scale=(0.85, 1), shear=[-2, 2], return_transform=True)
            images_aff, transform = random_affine(images_flip)
            logits_aff = self.model_base(images_aff)
            
            # hu.save_image('tmp1.png', images_aff[0])
            itransform = transform.inverse()
            logits_aff = kornia.geometry.transform.warp_affine(logits_aff, itransform[:,:2, :], dsize=(height, width))
            if flipped:
                logits_aff = flips.Hflip()(logits_aff)
            # hu.save_image('tmp2.png', images_recovered[0])

            
            # logits_flip = self.model_base(flips.Hflip()(images))

            loss = torch.mean(torch.abs(logits_aff-logits))
            points_aff = kornia.geometry.transform.warp_affine(points.float(), itransform[:,:2, :], dsize=(height, width), flags="nearest").long()
            # points_aff = points
            ind = points!=255
            if ind.sum() != 0:
                loss += F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')
                if flipped:
                    points_aff = flips.Hflip()(points_aff)
                # logits_flip =  self.model_base(flips.Hflip()(images))
                ind = points_aff!=255
                loss += F.binary_cross_entropy_with_logits(logits_aff[ind], 
                                        points_aff[ind].float().cuda(), 
                                        reduction='mean')
        elif loss_name == "elastic_cons_point_loss":
            points = points[:,None]
            ind = points!=255
            # self.vis_on_batch(batch, savedir_image='tmp.png')

            # POINT LOSS
            # loss = ut.joint_loss(logits, points[:,None].float().cuda(), ignore_index=255)
            # print(points[ind].sum())
            B, C, H, W = images.shape
            # ELASTIC TRANSFORM
            def norm_grid(grid):
                grid -= grid.min()
                grid /= grid.max()
                grid = (grid - 0.5) * 2
                return grid
            grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W))
            grid_x = grid_x.float().cuda()
            grid_y = grid_y.float().cuda()
            sigma=4
            alpha=34
            indices = torch.stack([grid_y, grid_x], -1).view(1, H, W, 2).expand(B, H, W, 2).contiguous()
            indices = norm_grid(indices)
            dx = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dx = torch.from_numpy(dx).cuda().float()
            dy = torch.from_numpy(dy).cuda().float()
            dgrid_x = grid_x + dx
            dgrid_y = grid_y + dy
            dgrid_y = norm_grid(dgrid_y)
            dgrid_x = norm_grid(dgrid_x)
            dindices = torch.stack([dgrid_y, dgrid_x], -1).view(1, H, W, 2).expand(B, H, W, 2).contiguous()
            dindices0 = dindices.permute(0, 3, 1, 2).contiguous().view(B*2, H, W)
            indices0 = indices.permute(0, 3, 1, 2).contiguous().view(B*2, H, W)
            iindices = torch.bmm(indices0, dindices0.pinverse()).view(B, 2, H, W).permute(0, 2, 3, 1)
            # indices_im = indices.permute(0, 3, 1, 2)
            # iindices = F.grid_sample(indices_im, dindices).permute(0, 2, 3, 1)
            aug = F.grid_sample(images, dindices)
            iaug = F.grid_sample(aug,iindices)


            # logits_aff = self.model_base(images_aff)
            # inv_transform = transform.inverse()
            
            import pylab
            def save_im(image, name):
                _images_aff = image.data.cpu().numpy()
                _images_aff -= _images_aff.min()
                _images_aff /= _images_aff.max()
                _images_aff *= 255
                _images_aff = _images_aff.transpose((1,2,0))
                pylab.imsave(name, _images_aff.astype('uint8'))
            save_im(aug[0], 'tmp1.png')
            save_im(iaug[0], 'tmp2.png')
            pass


        elif loss_name == 'lcfcn_loss':
            loss = 0.
  
            for lg, pt in zip(logits, points):
                loss += lcfcn_loss.compute_loss((pt==1).long(), lg.sigmoid())

                # loss += lcfcn_loss.compute_binary_lcfcn_loss(l[None], 
                #         p[None].long().cuda())

        elif loss_name == 'point_loss':
            points = points[:,None]
            ind = points!=255
            # self.vis_on_batch(batch, savedir_image='tmp.png')

            # POINT LOSS
            # loss = ut.joint_loss(logits, points[:,None].float().cuda(), ignore_index=255)
            # print(points[ind].sum())
            if ind.sum() == 0:
                loss = 0.
            else:
                loss = F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')
                                        
            # print(points[ind].sum().item(), float(loss))
        elif loss_name == 'att_point_loss':
            points = points[:,None]
            ind = points!=255

            loss = 0.
            if ind.sum() != 0:
                loss = F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')

                logits_flip = self.model_base(flips.Hflip()(images))
                points_flip = flips.Hflip()(points)
                ind = points_flip!=255
                loss += F.binary_cross_entropy_with_logits(logits_flip[ind], 
                                        points_flip[ind].float().cuda(), 
                                        reduction='mean')

        elif loss_name == 'cons_point_loss':
            points = points[:,None]
            
            logits_flip = self.model_base(flips.Hflip()(images))
            loss = torch.mean(torch.abs(flips.Hflip()(logits_flip)-logits))
            
            ind = points!=255
            if ind.sum() != 0:
                loss += F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')

                points_flip = flips.Hflip()(points)
                ind = points_flip!=255
                loss += F.binary_cross_entropy_with_logits(logits_flip[ind], 
                                        points_flip[ind].float().cuda(), 
                                        reduction='mean')

        return loss 

    def train_on_batch(self, batch):
        # add to seen images
        for m in batch['meta']:
            self.train_hashes.add(m['hash'])

        self.opt.zero_grad()

        images = batch["images"].cuda()
        
        

        # compute loss
        loss_name = self.exp_dict['model']['loss']
        if loss_name in ['joint_cross_entropy']:
            logits = self.model_base(images)
            # full supervision
            loss = self.compute_mask_loss(loss_name, images, logits, masks=batch["masks"].cuda())
        elif loss_name in ['point_loss', 'cons_point_loss', 'lcfcn_loss', 'affine_cons_point_loss', 'toponet']:
            logits = self.model_base(images)
            # point supervision
            loss = self.compute_point_loss(loss_name, images, logits, points=batch["points"].cuda())
        elif loss_name in ['multiscale_cons_point_loss']:
            logits = self.model_base(images, return_features=True)
            # point supervision
            loss = self.compute_point_loss(loss_name, images, logits, points=batch["points"].cuda())
        
        if loss != 0:
            loss.backward()
            if self.exp_dict['model'].get('clip_grad'):
                ut.clip_gradient(self.opt, 0.5)
            try:
                self.opt.step()
            except:
                self.opt.step(loss=loss)

        return {'train_loss': float(loss)}

    @torch.no_grad()
    def predict_on_batch(self, batch):
        self.eval()
        image = batch['images'].cuda()

        if hasattr(self.model_base, 'predict_on_batch'):
            return self.model_base.predict_on_batch(batch)
            s5, s4, s3, s2, se = self.model_base.forward(image)
            res = s2
            res = F.upsample(res, size=batch['meta'][0]['shape'],              
                         mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res > 0.5
            
        elif self.n_classes == 1:
            res = self.model_base.forward(image)
            if 'shape' in batch['meta'][0]:
                res = F.upsample(res, size=batch['meta'][0]['shape'],              
                            mode='bilinear', align_corners=False)
            res = (res.sigmoid().data.cpu().numpy() > 0.5).astype('float')
        else:
            self.eval()
            logits = self.model_base.forward(image)
            res = logits.argmax(dim=1).data.cpu().numpy()

        return res 

    def vis_on_batch(self, batch, savedir_image):
        image = batch['images']
        gt = np.asarray(batch['masks'], np.float32)
        gt /= (gt.max() + 1e-8)
        res = self.predict_on_batch(batch)

        image = F.interpolate(image, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        img_res = hu.save_image(savedir_image,
                     hu.denormalize(image, mode='rgb')[0],
                      mask=res[0], return_image=True)

        img_gt = hu.save_image(savedir_image,
                     hu.denormalize(image, mode='rgb')[0],
                      mask=gt[0], return_image=True)
        img_gt = models.text_on_image( 'Groundtruth', np.array(img_gt), color=(0,0,0))
        img_res = models.text_on_image( 'Prediction', np.array(img_res), color=(0,0,0))
        
        if 'points' in batch:
            pts = batch['points'][0].numpy().copy()
            pts[pts == 1] = 2
            pts[pts == 0] = 1
            pts[pts == 255] = 0
            img_gt = np.array(hu.save_image(savedir_image, img_gt/255.,
                                points=pts, radius=2, return_image=True))
        img_list = [np.array(img_gt), np.array(img_res)]
        hu.save_image(savedir_image, np.hstack(img_list))

    def val_on_loader(self, loader, savedir_images=None, n_images=0):
        self.eval()
        val_meter = metrics.SegMeter(split=loader.dataset.split)
        i_count = 0
        for i, batch in enumerate(tqdm.tqdm(loader)):
            # make sure it wasn't trained on
            for m in batch['meta']:
                assert(m['hash'] not in self.train_hashes)

            val_meter.val_on_batch(self, batch)
            if i_count < n_images:
                self.vis_on_batch(batch, savedir_image=os.path.join(savedir_images, 
                    '%d.png' % batch['meta'][0]['index']))
                i_count += 1

        
        return val_meter.get_avg_score()
        
    @torch.no_grad()
    def compute_uncertainty(self, images, replicate=False, scale_factor=None, n_mcmc=20, method='entropy'):
        self.eval()
        set_dropout_train(self)

        # put images to cuda
        images = images.cuda()
        _, _, H, W= images.shape

        if scale_factor is not None:
            images = F.interpolate(images, scale_factor=scale_factor)
        # variables
        input_shape = images.size()
        batch_size = input_shape[0]

        if replicate and False:
            # forward on n_mcmc batch      
            images_stacked = torch.stack([images] * n_mcmc)
            images_stacked = images_stacked.view(batch_size * n_mcmc, *input_shape[1:])
            logits = self.model_base(images_stacked)
            

        else:
            # for loop over n_mcmc
            logits = torch.stack([self.model_base(images) for _ in range(n_mcmc)])
            
            logits = logits.view(batch_size * n_mcmc, *logits.size()[2:])

        logits = logits.view([n_mcmc, batch_size, *logits.size()[1:]])
        _, _, n_classes, _, _ = logits.shape
        # binary do sigmoid 
        if n_classes == 1:
            probs = logits.sigmoid()
        else:
            probs = F.softmax(logits, dim=2)

        if scale_factor is not None:
            probs = F.interpolate(probs, size=(probs.shape[2], H, W))

        self.eval()

        if method == 'entropy':
            score_map = - xlogy(probs).mean(dim=0).sum(dim=1)

        if method == 'bald':
            left = - xlogy(probs.mean(dim=0)).sum(dim=1)
            right = - xlogy(probs).sum(dim=2).mean(0)
            bald = left - right
            score_map = bald


        return score_map 



class TrainMonitor:
    def __init__(self):
        self.score_dict_sum = {}
        self.n = 0

    def add(self, score_dict):
        for k,v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}

def set_dropout_train(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            module.train()

def xlogy(x, y=None):
    z = torch.zeros(())
    if y is None:
        y = x
    assert y.min() >= 0
    return x * torch.where(x == 0., z.cuda(), torch.log(y))