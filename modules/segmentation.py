import torch
import streamlit as st
import numpy as np

import mmcv
from mmcv.runner import load_checkpoint

from mmseg.models import build_segmentor
import cv2


import albumentations as A
from albumentations.pytorch import ToTensorV2

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
# Normalize = A.Normalize(mean=mean, std=std)
Totensor = ToTensorV2()

@st.cache
def load_model(cfg_path, ckpt_path):
    cfg = mmcv.Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 1
    
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
    
    model.PALETTE = [[0, 0, 0], [192, 0, 128], [0, 128, 192]]
    model.CLASSES = ['Background', 'printing', 'handwriting']
    
    return model

def preprocess_img(img):
    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 3

    img_meta = dict()
    img_meta['filename'] = ''
    img_meta['ori_filename'] = ''
    img_meta['ori_shape'] = img.shape

    img = cv2.resize(img, (img.shape[1]*2,img.shape[0]*2))
    # img = Normalize(image=img)['image']
    img = mmcv.imnormalize(img, np.array(mean), np.array(std), True)
    img_meta['img_shape'] = img.shape
    img_meta['pad_shape'] = img.shape
    
    scale_factor = [resize/ori for ori, resize in zip(img_meta['ori_shape'], img_meta['img_shape'][:2])]
    img_meta['scale_factor'] = np.array(scale_factor+scale_factor, dtype=np.float32)
    
    img = Totensor(image=img)['image']
    
    img_meta['flip'] = False
    img_meta['flip_direction'] = None
    
    img_meta['img_norm_cfg'] =  {'mean': np.array(mean, dtype=np.float32),
                                'std': np.array(std, dtype=np.float32),
                                'to_rgb': True}
    
    return img, img_meta

def slide_inference(model, images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    results = []
    for ori_img in images :
        torch.cuda.empty_cache()

        img, img_meta = preprocess_img(ori_img)
        img = img.unsqueeze(0).to(device)
        data = {'img_metas':[[img_meta]] , 'img':[img]}
        with torch.no_grad() :
            results.append(*model(return_loss=False, **data))
    return results, images

def ori_copy(ori_image, dst_image):
        
    h,w = dst_image.shape[:2]
    ori_image = cv2.resize(ori_image,(w,h))  
    
    dst_image_mask = np.zeros((h,w,1)).astype(np.uint8)
    dst_image_mask[dst_image[:,:,0] < 200] = 255 
    
    cv2.copyTo(ori_image, dst_image_mask, dst_image)


@st.cache(allow_output_mutation=True)
def seg_init():
    segmentor = load_model(cfg_path = './models/deeplabv3.py', 
                           ckpt_path = './checkpoints/best_mIoU_epoch_8.pth')

    return segmentor

  
@st.cache
def seg_image(model, images):    
    outputs, oris = slide_inference(model, images)

    results = list()
    for ori_image, output in zip(images, outputs):
        h, w = output.shape
        result = np.full((h,w,3), 255).astype(np.uint8)
        result[output == 1] = 0
        kernel = np.full((3,3), 1)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        # ori_copy(ori_image,result)
        results.append(result)
    
    return results