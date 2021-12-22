import torch
import streamlit as st
import numpy as np

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset

import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import mkdir


def slide_load_model(config_dir, checkpoint):
    cfg = mmcv.Config.fromfile(config_dir)
    
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    cfg.model.train_cfg = None

    # slide window
    cfg.model.test_cfg.mode = 'slide'
    cfg.model.test_cfg.stride = (200,200)
    cfg.model.test_cfg.crop_size = (384,384)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE
    
    return MMDataParallel(model, device_ids=[0]), cfg


def slide_inference(model, cfg, images):
    cfg.data.test.img_dir = './crop_images/'
    cfg.data.test.img_suffix = '.png'
    cfg.data.test.ann_dir = None
    
    mkdir(cfg.data.test.img_dir)
    for idx, image in enumerate(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(f'{cfg.data.test.img_dir}{str(idx).zfill(2)}.png', image)

    test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[2.0],
        flip=False,
        transforms=[
            dict(type='Resize',keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
    ]
    cfg.data.test.pipeline = test_pipeline
    
    dataset = build_dataset(cfg.data.test)
    
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            shuffle=False)

    torch.cuda.empty_cache()

    model.eval()
    results,ori = [],[]
    for data in data_loader:
        with torch.no_grad():
            results.append(*model(return_loss=False, **data))
            ori.append(data['img'][0])
    os.system(f'rm -r {cfg.data.test.img_dir}')
    return results, ori


def ori_copy(ori_image, output):
    image_mask = np.zeros(output.shape).astype(np.uint8)
    x, y, _ = np.where(output == 0)
    for x_, y_ in zip(x, y):
        image_mask[x_, y_, :] = 255

    cv2.copyTo(ori_image, image_mask, output)
    return output


@st.cache
def seg_image(images):    
    config_dir = './checkpoints/deeplabv3.py'
    checkpoint = './checkpoints/best_mIoU_epoch_8.pth'
    model, cfg = slide_load_model(config_dir, checkpoint)
    outputs, oris = slide_inference(model, cfg, images)

    results = list()
    for ori_image, output in zip(images, outputs):
        h, w = output.shape
        result = np.full((h,w,3), 255).astype(np.uint8)
        result[output == 1] = 0
        kernel = np.full((4,4), 1)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        results.append(result)
    
    return results