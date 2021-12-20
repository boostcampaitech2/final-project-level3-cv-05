import numpy as np
import streamlit as st
import torch
from models.AttentionRes_G import AttentionRes_G
from models.CycleGAN_G import CycleGAN_G
import albumentations as A
import torch.nn as nn
import cv2

@st.cache
def load_attgan_model(model_file):
    net = AttentionRes_G(3, 3, 64, n_blocks=9)
    load_path = './checkpoints/'+model_file
    state_dict = torch.load(load_path, map_location=str('cuda'))
    net.load_state_dict(state_dict)
    return net

#GAN
@st.cache
def GAN_image(images):
    ''' Erase Handwriting in image 
    Parameters:
        images : PIL Image Type
    '''
    # load model
    gan_model = load_attgan_model('attentiongan.pth')
    # data transform
    load_size = 512
    img_transform = A.Compose([
        A.Resize(load_size, load_size, 2),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        A.pytorch.ToTensorV2()])
    inputs = list()
    for image in images:
        inputs.append(img_transform(image=image)['image'])
    inputs = torch.stack(inputs, 0)
    with torch.no_grad():
        output = gan_model(inputs)
        outputs = tensor2im(output)
    return outputs

@st.cache
def load_cyclegan_model(model_file):
    net = CycleGAN_G(2, 2, 8, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    load_path = './checkpoints/'+model_file
    state_dict = torch.load(load_path, map_location=str('cuda'))
    net.load_state_dict(state_dict)
    return net

@st.cache
def Inpainting_image(ori_images, target_images):
    ''' Inpainting the result of segmentation 
    Parameters:
        ori_images : list of original images (before segmentation) (numpy array type)
        target_images : list of images to inpaint (after segmentation) (numpy array type)
    '''
    
    # load model
    gan_model = load_cyclegan_model('latest_net_G_A.pth')
    # data transform
    load_size = [256,768]
    img_transform = A.Compose([
        A.Resize(load_size[0], load_size[1], 2),
        A.ToGray(always_apply=True),
        A.Normalize((0.5,), (0.5,)),
        A.pytorch.ToTensorV2()])
    
    inputs = list()
    input_shapes = list()
    for idx, ori_image in enumerate(ori_images):
        input_shapes.append(ori_image.shape)
        ori_img = img_transform(image=cv2.cvtColor(ori_image, cv2.COLOR_RGB2GRAY))['image']
        target_img = img_transform(image=cv2.cvtColor(target_images[idx], cv2.COLOR_RGB2GRAY))['image']
        concat_img = torch.cat([ori_img, target_img])
        inputs.append(concat_img)
    inputs = torch.stack(inputs, 0)
    
    with torch.no_grad():
        output = gan_model(inputs)
        output = torch.mean(output, dim=1, keepdim=True)
        outputs = tensor2im(output, input_shapes)


    return outputs


def tensor2im(input_image, input_im_shape, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    images = []
    image_numpys = input_image.data.cpu().float().numpy()
    for im_idx, image_numpy in enumerate(image_numpys):
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_shape = (input_im_shape[im_idx][1], input_im_shape[im_idx][0])
        image_numpy = cv2.resize(image_numpy, dsize=image_shape, interpolation=cv2.INTER_AREA)
        images.append(image_numpy.astype(imtype))
    return images