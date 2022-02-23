from .diverseDepth.diverse_depth_model import RelDepthModel
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import os,dill
import argparse
import numpy as np
import torch
from collections import OrderedDict


def load_weights(model):
    """
    Load checkpoint.
    """
    #loc = os.path.dirname(os.path.abspath(__file__))
    #print(loc)
    #weights= os.path.join(loc, "diverseDepth/model.pth")
    #weights=  "/content/drive/MyDrive/weights/depthEstimate.pth"

    weights=  "./weights/depthEstimate.pth"
    checkpoint = torch.load(weights, map_location=lambda storage, loc: storage, pickle_module=dill)
    model_state_dict_keys = model.state_dict().keys()
    checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")

    if all(key.startswith('module.') for key in model_state_dict_keys):
        model.module.load_state_dict(checkpoint_state_dict_noprefix)
    else:
        model.load_state_dict(checkpoint_state_dict_noprefix)
    del checkpoint
    return model
    torch.cuda.empty_cache()


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def init_Depth_model():
    depth_model = RelDepthModel()
    depth_model.eval()

    # load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model=load_weights(depth_model)
    depth_model.to(device)
    return depth_model

def get_depth(model,batch):
    transforms1 = transforms.Compose([transforms.Resize((385, 385), transforms.InterpolationMode.BICUBIC),
                                        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
    transforms2 = transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC)
    batch = transforms1(batch)
    pred_depth = model.inference(batch)
    pred_depth = transforms2(pred_depth)
    return pred_depth
  

def get_single_depth(img):
    depth_model = init_Depth_model()
    
    rgb = cv2.imread(img)
    rgb_c = rgb[:, :, ::-1].copy()
    rgb_c = transforms.ToTensor()(rgb_c)[None,:,:,:]
    depthMap = get_depth(depth_model,rgb_c)
    pred_depth_ori = depthMap.cpu().numpy().squeeze()


    #plt.imsave(img[:-4]+'-depth.png', pred_depth_ori, cmap='rainbow')
    cv2.imshow("img", (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
    cv2.waitKey(0) 
    

#if __name__ == '__main__':
    #img = os.path.join(os.path.dirname(os.path.abspath(__file__)),"img1.jpg")
    #get_single_depth(img)
    
    #x = torch.zeros(16, 3, 256, 256, dtype=torch.float, requires_grad=False)
    #depth_model = init_Depth_model()
    #depthMap = get_depth(depth_model,x)
