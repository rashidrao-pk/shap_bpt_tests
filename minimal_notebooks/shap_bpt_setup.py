import numpy as np
import cv2
import numpy as np
import torch
import torchvision
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

from skimage.filters import gaussian

from shap_bpt_setup import *

#=================================================================================================================
def saliency_to_auc(nu, heatmap, f_S, f_0, predicted_cls, batch_size=4, method='del', num_samples=101, 
                rule='trapezoid'):
    assert isinstance(heatmap, np.ndarray)
    assert len(heatmap.shape)==2 and np.issubdtype(heatmap.dtype, np.floating)

    # nu_max = max(f_S, f_0)
    # nu_min = min(f_S, f_0)

    xs, ys, ms, masks, qs = [], [], [], [], []
    for i, value in enumerate(np.linspace(start=1.0, stop=0.0, num=num_samples)):
        if method=='del':
            epsilon = (1 if value==0.0 else 0)
            q = (np.quantile(heatmap, q=value) - epsilon)
            m = heatmap <= q
            nx = (1.0 - np.sum(m) / m.size)
        elif method=='ins':
            epsilon = (1 if value==1.0 else 0)
            q = (np.quantile(heatmap, q=value) + epsilon)
            m = heatmap >= q
            nx = (np.sum(m) / m.size)
        else:
            raise Exception()
            
        # add a new datapoint on the curve
        if len(xs)==0 or nx != xs[-1]: 
            assert m.dtype==bool and len(m.shape)==2
            xs.append(nx)
            masks.append(m)
            ms.append(np.sum(heatmap[m]))
            qs.append(q)

        # evaluate the characteristic function
        if len(masks) >= batch_size or (len(masks)>0 and i==(num_samples-1)):
            y = nu(np.array(masks))[:, predicted_cls]
            ys.extend(y)
            masks = []

    assert len(masks)==0    
    xs, ys = np.array(xs), np.array(ys)
    assert(len(xs) == len(ys))

    # compute considering under/over shoots
    if f_S > f_0:
        overshoot_max = np.maximum(0, ys - f_S) # overshoot for values exceeding the maximum f(S)
        overshoot_min = np.maximum(0, f_0 - ys) # overshoot for values below the minimum f(0)
    else: # f(S) < f(0)
        overshoot_max = np.maximum(0, ys - f_0) # overshoot for values exceeding the maximum f(0)
        overshoot_min = np.maximum(0, f_S - ys) # overshoot for values below the minimum f(S)

    # clip ys, no oveshoots
    y_clipped = np.clip(ys, min(f_S, f_0), max(f_S, f_0))
    # adjust ys with the overshoot. Clip it inside the admitted range
    y_adjusted = np.clip(ys - 2*overshoot_max + 2*overshoot_min, min(f_S, f_0), max(f_S, f_0))

    # rebase to f(0)
    if f_S > f_0:
        flipped = False
        ys = ys - f_0 
        y_clipped = y_clipped - f_0 
        y_adjusted = y_adjusted - f_0
    else: # f(S) < f(0)
        flipped = True
        ys = f_0 - ys 
        y_clipped = f_0 - y_clipped 
        y_adjusted = f_0 - y_adjusted

    # rescaling
    ys_rescaled = ys / abs(f_S - f_0)
    y_clipped_rescaled = y_clipped / abs(f_S - f_0)
    y_adjusted_rescaled = y_adjusted / abs(f_S - f_0)

    auc, auc_r, auc_mae, auc_mse, auc_adj, auc_adjr, auc_clip, auc_clipr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    curve_range = range(1, len(xs)) if rule=='trapezoid' else range(len(xs))

    # compute the area under the curve with the midpoint Riemann sum (i.e. the trapezoidal rule)
    for i in curve_range:
        if rule=='trapezoid':
            delta_x = abs(xs[i] - xs[i-1])
            assert delta_x > 0
            y_mid   =         0.5*(ys[i-1] + ys[i])
            y_r_mid =         0.5*(ys_rescaled[i-1] + ys_rescaled[i])
            err_mid = y_mid - 0.5*(ms[i-1] - ms[i])
            y_clip_mid =       0.5*(y_clipped[i-1] + y_clipped[i])
            y_clipr_mid =      0.5*(y_clipped_rescaled[i-1] + y_clipped_rescaled[i])
            y_adj_mid =       0.5*(y_adjusted[i-1] + y_adjusted[i])
            y_adjr_mid =      0.5*(y_adjusted_rescaled[i-1] + y_adjusted_rescaled[i])
        else: # rectangles
            delta_x = 1.0/num_samples if i==len(xs)-1 else abs(xs[i+1] - xs[i])
            assert delta_x > 0
            y_mid   =         ys[i]
            y_r_mid =         ys_rescaled[i]
            err_mid = y_mid - ms[i]
            y_clip_mid =       y_clipped[i]
            y_clipr_mid =      y_clipped_rescaled[i]
            y_adj_mid =       y_adjusted[i]
            y_adjr_mid =      y_adjusted_rescaled[i]


        auc += abs(delta_x * y_mid) # base * height
        auc_r += abs(delta_x * y_r_mid) # base * height
        # auc_eff += abs(delta_x * err_mid) # base * height
        auc_mae += abs(delta_x * err_mid) # base * height
        auc_mse += abs(delta_x * (err_mid**2)) # base * height^2
        auc_clip += abs(delta_x * y_clip_mid)
        auc_clipr += abs(delta_x * y_clipr_mid)
        auc_adj += abs(delta_x * y_adj_mid)
        auc_adjr += abs(delta_x * y_adjr_mid)

    return {'xs':xs, 'ms':ms, 'qs':qs, 
            'f_0':f_0, 'f_S':f_S, 'flipped':flipped, 
            'ys':ys, 'ysr':ys_rescaled,
            'y_clip':y_clipped, 'y_clipr':y_clipped_rescaled, 
            'y_adj':y_adjusted, 'y_adjr':y_adjusted_rescaled, 
            'method':method, 'predicted_cls':predicted_cls,
            'auc':auc, 'auc_r':auc_r,
            'auc_mae':auc_mae, 'auc_mse':auc_mse, 'auc_rmse':np.sqrt(auc_mse), 
            'auc_clip':auc_clip, 'auc_clipr':auc_clipr,
            'auc_adj':auc_adj, 'auc_adjr':auc_adjr}


#### Load image and background images

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
resnet50_preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
)

import urllib.request
import json

def get_imagenet_classes(url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"):
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())

    class_names = [v[1] for v in data.values()]
    print("Number of ImageNet classes:", len(class_names))
    return class_names    
#--------------------------------------------------------------------------------------------------------
def load_image(ds_path='D:/DS/ImageNet/ImageNet/ILSVRC2012_img_val//',fname='ILSVRC2012_val_00000774',bg_type=None,device='cpu'):
    image_to_explain         = cv2.resize(cv2.imread(f'{ds_path}/{fname}.JPEG', cv2.IMREAD_COLOR),[224,224])[:,:,::-1]
    image_to_explain_tensor  = resnet50_preprocess(image_to_explain.astype(np.float32)/255.0).to(device)

    # image_to_explain_tensor = image_to_explain_preproc.to(device)
    np.random.seed(0)

    bkgnd0 = np.full_like(image_to_explain, 0)
    bkgnd1 = np.full_like(image_to_explain, 127)
    bkgnd2 = np.full_like(image_to_explain, 255)
    bkgnd3 = gaussian(image_to_explain, 8, channel_axis=-1)*255
    bkgnd4 = np.clip(np.random.normal(128, 128, size=image_to_explain.shape), 0, 255).astype(np.uint8)
    bkgnd4 = (gaussian(bkgnd4, 2.0, channel_axis=-1) * 255).astype(np.uint8)
    if bg_type=='black':
        background_image_set = np.array([bkgnd0])
    elif bg_type=='gray':
        background_image_set = np.array([bkgnd1])
    elif bg_type=='white':
        background_image_set = np.array([bkgnd2])
    elif bg_type=='blurred':
        background_image_set = np.array([bkgnd3])
    elif bg_type=='noise':
        background_image_set = np.array([bkgnd4])
    elif bg_type=='full':
        background_image_set = np.array([bkgnd0, bkgnd1, bkgnd2, bkgnd3, bkgnd4])

    background_image_preproc_set = [resnet50_preprocess(bkgnd.astype(np.float32)/255.0)
                                        for bkgnd in background_image_set]

    background_tensors = torch.cat([torch.unsqueeze(bk_p, dim=0) 
                                    for bk_p in background_image_preproc_set]).to(device)
    return image_to_explain,image_to_explain_tensor,background_image_set,background_tensors

def f_masked_resnet(masks):
    N,H,W = masks.shape
    B = background_tensors.shape[0]
    masks_tensor = torch.tensor(np.array(masks)).to(device)                     # N boolean masks with size W*H
    masks_tensor = torch.reshape(masks_tensor, (N,1,H,W))                       # N*H*W    -> N*1*H*W
    masks_tensor = torch.tile(masks_tensor, dims=(1,3,B,1)).reshape(B*N,3,H,W)  # N*1*H*W  -> NB*3*H*W
    Xf = torch.tile(image_to_explain_tensor, dims=(N*B,1,1,1))                  # 3*H*W    -> NB*3*H*W
    Xb = torch.tile(background_tensors, dims=(N,1,1,1))                         # 3*H*W    -> NB*3*H*W
    X = torch.where(masks_tensor, Xf, Xb) # T=Xf, F=Xb
    result = f(X)
    del X, Xb, Xf, masks_tensor
    result = result.reshape((-1, B, result.shape[1]))
    return np.mean(result, axis=1)

def f_masked(masks):
    global params
    return f_masked_resnet(masks)
# shap_bpt_tests = shap_bpt_tests()

#=================================================================================================================
def load_image_to_explain(f,model,fname=None, bg_type=None,device='cpu'):

    image_to_explain,_,background_image_set,background_tensors = load_image(fname=fname,bg_type=bg_type)
    
    h,w,_ = image_to_explain.shape
    # Foreground image to be explained   
    predicted_fS = f(torch.unsqueeze(resnet50_preprocess(image_to_explain.astype(np.float32)/255.0).to(device), dim=0))[0]
    sorted_classes = np.flip(np.argsort(predicted_fS))
    predicted_cls = sorted_classes[0]
    f_S = float(predicted_fS[predicted_cls])
    #####################
    predicted_f0 = [f(torch.unsqueeze(resnet50_preprocess(bkgnd.astype(np.float32)/255.0).to(device), dim=0))[0] for bkgnd in background_image_set]
    predicted_f0 = np.mean(predicted_f0,axis=0)
    f_0          = float(predicted_f0[predicted_cls])
    return predicted_fS,predicted_cls, f_S, predicted_f0, f_0

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if ('mps' in dir(torch.backends)) and torch.backends.mps.is_available() else torch.device("cpu")