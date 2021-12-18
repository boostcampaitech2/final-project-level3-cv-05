import numpy as np
import streamlit as st
import torch
from models.AttentionRes_G import AttentionRes_G
import albumentations as A

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

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    images = []
    image_numpys = input_image.data.cpu().float().numpy()
    for image_numpy in image_numpys:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        images.append(image_numpy.astype(imtype))
    return images