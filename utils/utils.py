import torch
import numpy as np
from skimage.color import rgb2lab, lab2rgb

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count,self.avg,self.sum = [0.]*3
        
    def update(self,val,count=1):
        self.count +=count
        self.sum += count*val
        self.avg =self.sum / self.count
        
def create_loss_meters(train = True):
    if train:
      loss = AverageMeter()
      
      return {'loss':loss}

    else:
      psnr = AverageMeter()
      ssim = AverageMeter()
      
      return {
              'psnr':psnr,
              'ssim':ssim
          }

def update_losses(model,loss_meter_dict,count):
    for loss_name,loss_meter in loss_meter_dict.items():
        loss = getattr(model,loss_name)
        if "loss" in loss_name:
          loss_meter.update(loss.item(),count=count)
        else:
          loss_meter.update(loss,count=count)
        
def lab_to_rgb(L,ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb=lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs,axis=0)


        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")