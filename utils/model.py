import sys
sys.path.append("..")

from detection import load_model, get_crop_location, draw_from_crop_locations, crop_from_crop_locations
from models.AttentionRes_G import AttentionRes_G
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision.transforms as transforms
import numpy as np

import streamlit as st

@st.cache
def load_attgan_model(model_file):
    net = AttentionRes_G(3, 3, 64, n_blocks=9)
    load_path = './checkpoints/'+model_file
    state_dict = torch.load(load_path, map_location=str('cuda'))
    net.load_state_dict(state_dict)
    return net

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    images = []
    image_numpys = input_image.data.cpu().float().numpy()
    for image_numpy in image_numpys:
        # image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        images.append(image_numpy.astype(imtype))
    return images

#Object Detection
@st.cache
def OD_image(image):
    ''' Crop uncorrect problem
    Parameters:
        image : PIL Image Type
    '''
    detector = load_model(cfg_path = "./checkpoints/yolov3_config.py", 
                    ckpt_path = "./checkpoints/yolov3_weight.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = detector.to(device)
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    locations = get_crop_location(detector, image)   # [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
    drawed_img = draw_from_crop_locations(image, locations)
    croped_img = crop_from_crop_locations(image, locations)
    return drawed_img, croped_img


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
    # img_transform = transforms.Compose(
    #     [transforms.Resize([load_size,load_size], Image.BICUBIC),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_transform = A.Compose(
        [A.Resize(load_size, load_size, 2),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensorV2()])
    inputs = []
    for image in images:
        inputs.append(img_transform(image=image)['image'])
    inputs = torch.stack(inputs, 0)
    # af_transform = img_transform(image)
    # c,w,h = af_transform.shape
    # af_transform = np.reshape(af_transform, (1,c,w,h)) # convert to batch form
    # forward and tensor to image
    with torch.no_grad():
        output = gan_model(inputs)
        outputs = tensor2im(output)
    return outputs