import os
import numpy as np
import cv2

import torch
import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset

from tqdm import tqdm

DEFAULT_PATH = '/opt/ml/project/mmseg/version/07'

def mkdir(dir_):
    if not os.path.isdir(dir_):
        os.mkdir(dir_)

def move_images():
    IMAGE_PATH = f'{DEFAULT_PATH}/images/train'
    ANNOTATION_PATH = f'{DEFAULT_PATH}/annotations/train'
    INFERENCE_PATH = f'{DEFAULT_PATH}/inference'

    mkdir(f'{INFERENCE_PATH}/combine_img')
    mkdir(f'{INFERENCE_PATH}/ori_combine_img')
    mkdir(f'{INFERENCE_PATH}/final_img')
    mkdir(f'{INFERENCE_PATH}/resize_img')
    mkdir(f'{INFERENCE_PATH}/ori_copy')

    dataset_length = len(os.listdir(f'{DEFAULT_PATH}/images/train'))

    for idx in tqdm(range(dataset_length)):
        image_path = os.path.join(IMAGE_PATH, str(idx).zfill(3))
        annotation_path = os.path.join(ANNOTATION_PATH, str(idx).zfill(3))
        inference_path = os.path.join(INFERENCE_PATH, str(idx).zfill(3))
        mkdir(image_path)
        mkdir(annotation_path)
        mkdir(inference_path)

        os.system(f'mv {os.path.join(IMAGE_PATH, str(idx).zfill(4))}* {image_path}')
        os.system(f'mv {os.path.join(ANNOTATION_PATH, str(idx).zfill(4))}* {annotation_path}')
    


def combine_window(index):
    PATH = f'{DEFAULT_PATH}/inference/{str(index).zfill(3)}'
    #PATH = f'{DEFAULT_PATH}/images/train/{str(index).zfill(3)}'

    images = sorted(os.listdir(PATH))
    combine_image = np.zeros((384*3, 384*3))
    points = [(0, 0), (0, 384), (0, 384*2), (384, 0), (384, 384), (384, 384*2), (384*2, 0), (384*2, 384), (384*2, 384*2)]
    for idx in range(9):
        seg_img = cv2.imread(os.path.join(PATH, images[idx]), cv2.IMREAD_GRAYSCALE)
        point = points[idx]
        combine_image[point[0]:point[0]+384,point[1]:point[1]+384] = seg_img

    img = np.full(combine_image.shape, 255)
    img[combine_image == 1] = 0

    cv2.imwrite(f'{DEFAULT_PATH}/inference/combine_img/{str(index).zfill(3)}.png', img)


def ori_combine_window(index):
    PATH = f'{DEFAULT_PATH}/images/train/{str(index).zfill(3)}'

    images = sorted(os.listdir(PATH))
    patch_h, patch_w = 384, 384
    combine_image = np.zeros((patch_h*3, patch_w*3))
    points = [(0, 0), (0, patch_w), (0, patch_w*2), (patch_h, 0), (patch_h, patch_w), 
          (patch_h, patch_w*2), (patch_h*2, 0), (patch_h*2, patch_w), (patch_h*2, patch_w*2)]

    for idx in range(9):
        seg_img = cv2.imread(os.path.join(PATH, images[idx]), cv2.IMREAD_GRAYSCALE)
        point = points[idx]
        combine_image[point[0]:point[0]+384,point[1]:point[1]+384] = seg_img    

    cv2.imwrite(f'{DEFAULT_PATH}/inference/ori_combine_img/{str(index).zfill(3)}.png', combine_image)


def load_model(config_dir, checkpoint):
    cfg = mmcv.Config.fromfile(config_dir)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    cfg.model.train_cfg = None
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

def inference(model, cfg, index):
    cfg.data.test.img_dir = f'{DEFAULT_PATH}/images/train/{str(index).zfill(3)}'
    cfg.data.test.ann_dir = f'{DEFAULT_PATH}/annotations/train/{str(index).zfill(3)}'
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=9,
            workers_per_gpu=1,
            shuffle=False)

    torch.cuda.empty_cache()

    model.eval()
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        for idx in range(len(result)):
            cv2.imwrite(f'{DEFAULT_PATH}/inference/{str(index).zfill(3)}/{str(idx).zfill(2)}.png', result[idx])
            

def post_processing(index):
    image_path = f'{DEFAULT_PATH}/inference/combine_img/{str(index).zfill(3)}.png'
    img = cv2.imread(image_path, 0)
    kernel = np.full((3,3), 1)
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f'{DEFAULT_PATH}/inference/final_img/{str(index).zfill(3)}.png', result)


def post_resize(index):
    image_path = f'{DEFAULT_PATH}/inference/final_img/{str(index).zfill(3)}.png'
    img = cv2.imread(image_path)
    
    original_path = f'/opt/ml/dummy/clean_code/background/{str(index).zfill(5)}.png'
    ori_img = cv2.imread(original_path)
    
    result = cv2.resize(img, dsize=(ori_img.shape[1], ori_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(f'{DEFAULT_PATH}/inference/resize_img/{str(index).zfill(3)}.png', result)
    

def ori_copy(index):
    image_path = f'{DEFAULT_PATH}/inference/resize_img/{str(index).zfill(3)}.png'
    img = cv2.imread(image_path)

    original_path =  f'{DEFAULT_PATH}/inference/ori_combine_img/{str(index).zfill(3)}.png'
    ori_img = cv2.imread(original_path)
    
    ori_img = cv2.resize(ori_img, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    img_mask = np.zeros((img.shape[0], img.shape[1],1)).astype(np.uint8)
            
    x,y,_ = np.where(img < 200)
    for x_,y_ in zip(x,y) :
        img_mask[x_,y_,:] = 255 

    cv2.copyTo(ori_img, img_mask, img)

    cv2.imwrite(f'{DEFAULT_PATH}/inference/ori_copy/{str(index).zfill(3)}.png', img)

if __name__ == "__main__":
    config_dir = '/opt/ml/mmsegmentation/tools/work_dirs/deeplabv3plus/deeplabv3plus.py'
    checkpoint = '/opt/ml/mmsegmentation/tools/work_dirs/deeplabv3plus/best_mIoU_epoch_13.pth'

    move_images() 
    model, cfg = load_model(config_dir, checkpoint)

    dataset_length = len(os.listdir(f'{DEFAULT_PATH}/images/train'))
    for index in tqdm(range(dataset_length)):
        inference(model, cfg, index)
        combine_window(index)
        post_processing(index)
        post_resize(index)
        ori_combine_window(index)
        ori_copy(index)