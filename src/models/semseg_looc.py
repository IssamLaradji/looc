import os, pprint, tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu 
from haven import haven_img as hi
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import fcn8_vgg16
from . import semseg_counting
from . import lcfcn 
from src import utils as ut
from . import attention_network
from skimage import morphology as morph
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage

from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
import pylab as plt

class SemSegLooc(semseg_counting.SemSegCounting):
    def __init__(self, exp_dict):
        super().__init__(exp_dict)
        
        self.att_model = attention_network.AttModel(exp_dict)

    def train_on_batch(self, batch, **extras):
        
        self.train()

        images = batch["images"].cuda()
        counts = float(batch["counts"][0])

        logits = self.model_base(images)
        if self.exp_dict['model'].get('loss') == 'lcfcn':
            loss = lcfcn.compute_lcfcn_loss(logits, 
                            batch["points"].cuda(),
                            None
                            )
            probs = F.softmax(logits, 1); 
            mask = probs.argmax(dim=1).cpu().numpy().astype('uint8').squeeze()*255
            
            # img_mask=hu.save_image('tmp2.png', 
            #             hu.denormalize(images, mode='rgb'), mask=mask, return_image=True)
            # hu.save_image('tmp2.png',np.array(img_mask)/255. , radius=3,
            #                 points=batch["points"])

        elif self.exp_dict['model'].get('loss') == 'glance':
            pred_counts = logits[:,1].mean()
            loss = F.mse_loss(pred_counts.squeeze(),
                               counts.float().squeeze())
                               
        elif self.exp_dict['model'].get('loss') == 'att_lcfcn':
            probs = logits.sigmoid() 
            
            # get points from attention
            att_dict = self.att_model.get_attention_dict(images_original=
                torch.FloatTensor(hu.denormalize(batch['images'], mode='rgb')), counts=batch['counts'][0],
                            probs=probs.squeeze(),
                return_roi=True)
            if 1:
                blobs = lcfcn.get_blobs(probs.squeeze().detach().cpu().numpy())
                org_img = hu.denormalize(images.squeeze(), mode='rgb')
                rgb_labels = label2rgb(blobs,hu.f2l(org_img),  bg_label=0, bg_color=None, )
                res1 = mark_boundaries(rgb_labels,  blobs)
                img_mask=hu.save_image('tmp2.png', 
                            org_img, return_image=True)
                res2 = hu.save_image('tmp.png',np.array(img_mask)/255. , 
                                points=att_dict['points'], radius=1, return_image=True)
                
                hu.save_image('tmp_blobs.png', np.hstack([res1, np.array(res2)/255.]))

            loss = lcfcn.compute_loss(probs=probs, 
                            # batch["points"].cuda(),
                            points=att_dict['points'].cuda(),
                            roi_mask=att_dict['roi_mask']
                            )
            # loss += .5 * F.cross_entropy(logits, 
            #             torch.from_numpy(1 - 
            #                 att_dict['mask_bg']).long().cuda()[None], 
            #             ignore_index=1)

        self.opt.zero_grad()
        loss.backward()
        if self.exp_dict['optimizer'] == 'sps':
            self.opt.step(loss=loss)
        else:
            self.opt.step()
   
        return {"train_loss":float(loss)}

    @torch.no_grad()
    def vis_on_batch(self, batch, savedir_image, save_preds=False):
        self.eval()   
        images = batch["images"].cuda()
        counts = float(batch["counts"][0])

        logits = self.model_base(images)

        probs = logits.sigmoid() 
            
        # get points from attention
        att_dict = self.att_model.get_attention_dict(images_original=
            torch.FloatTensor(hu.denormalize(batch['images'], mode='rgb')), counts=batch['counts'][0],
                        probs=probs.squeeze(),
            return_roi=True)

        blobs = lcfcn.get_blobs(probs.squeeze().detach().cpu().numpy())
        org_img = hu.denormalize(images.squeeze(), mode='rgb')
        rgb_labels = label2rgb(blobs,hu.f2l(org_img),  bg_label=0, bg_color=None, )
        res1 = mark_boundaries(rgb_labels,  blobs)

        if att_dict['roi_mask'] is not None:
            img_mask=hu.save_image('tmp2.png', 
                        org_img, 
                        mask=att_dict['roi_mask']==1, return_image=True)
            res2 = hu.save_image('tmp.png',np.array(img_mask)/255. , 
                            points=att_dict['points'], radius=1, return_image=True)
            
            os.makedirs(os.path.dirname(savedir_image), exist_ok=True)
            # plt.savefig(savedir_image.replace('.jpg', '.png')
            hu.save_image(savedir_image.replace('.jpg', '.png'), np.hstack([res1, np.array(res2)/255.]))
        