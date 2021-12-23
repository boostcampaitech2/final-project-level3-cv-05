import argparse
import os
import cv2
import torch
import numpy as np
from copy import deepcopy

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import LongestMaxSize

import streamlit as st
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector


Resize = LongestMaxSize(1024)
mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
Normalize = A.Normalize(mean=mean, std=std)
Totensor = ToTensorV2()

@st.cache
def load_model(cfg_path, ckpt_path):
    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    samples_per_gpu = 1
    cfg.model.train_cfg = None
    cfg.data.workers_per_gpu = 1
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
    model.CLASSES = ['correct', 'uncorrect']
    return model

def preprocess_img(img):
    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 3

    img_meta = dict()
    img_meta['filename'] = ''
    img_meta['ori_filename'] = ''
    img_meta['ori_shape'] = img.shape

    img = Resize(image=img)['image']
    img = Normalize(image=img)['image']
    img_meta['img_shape'] = img.shape
    
    # size_divisor pad
    if not img.shape[0]%32 == 0:
        img = np.pad(img, ((0,32-img.shape[0]%32),(0,0),(0,0)), 'constant', constant_values=0)
    if not img.shape[1]%32 == 0:
        img = np.pad(img, ((0,0),(0,32-img.shape[1]%32),(0,0)), 'constant', constant_values=0)
    img_meta['pad_shape'] = img.shape
    
    scale_factor = [resize/ori for ori, resize in zip(img_meta['ori_shape'], img_meta['img_shape'][:2])]
    img_meta['scale_factor'] = np.array(scale_factor+scale_factor)
    img = Totensor(image=img)['image']
    
    img_meta['flip'] = False
    img_meta['flip_direction'] = None
    
    img_meta['img_norm_cfg'] =  {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32),
                                'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32),
                                'to_rgb': True}

    return img, img_meta

def get_crop_location(model, img, uncorrect=True, score_thresh=0.3):
    """

    return crop_locations : [[x1,y1,x2,y2], ...]
    """
    img, img_meta = preprocess_img(img)
    device = next(model.parameters()).device
    img = img.unsqueeze(0).to(device)
    output = model.simple_test(img, [img_meta], rescale= True)

    if uncorrect:
        output = output[0][1]
    else:
        output = output[0][0]

    crop_locations = []
    for location in output:
        if location[4] > score_thresh:
            crop_locations.append([int(point) for point in location[:4]])

    return crop_locations

def draw_from_crop_locations(img, crop_locations, color=(255,0,0), thickness=3):
    new_img = deepcopy(img)
    for x1,y1,x2,y2 in crop_locations:
        print(x1,y1,x2,y2)
        new_img = cv2.rectangle(new_img, (x1,y1), (x2,y2), color, thickness)
    return new_img

def crop_from_crop_locations(img, crop_locations):
    crop_images = []
    for x1,y1,x2,y2 in crop_locations:
        crop_images.append(img[y1:y2,x1:x2])
    return crop_images

#Object Detection
@st.cache(allow_output_mutation=True)
def det_init():
    detector = load_model(cfg_path = "./models/yolov3_config.py", 
                    ckpt_path = "./checkpoints/yolov3_weight.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = detector.to(device)

    return detector


@st.cache
def det_image(detector, image):
    ''' Crop uncorrect problem
    Parameters:
        image : PIL Image Type
    '''
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    locations = get_crop_location(detector, image)   # [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
    drawed_img = draw_from_crop_locations(image, locations)
    croped_img = crop_from_crop_locations(image, locations)
    return drawed_img, croped_img, locations
