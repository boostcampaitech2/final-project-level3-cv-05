import torch

def get_accuracy(real_img, fake_img, threshold=0.95):
    '''
    Parameters:
        real_img(tensor) - tensor type image(gray scale)
        fake_img(tensor) - tensor type image(gray scale)
        threshold(float) - Threshold value of pixels to be treated as white pixels
    
    Return:
        custom accuarcy
    '''
    
    # mask pixels -> black : True, white : False
    real_img_mask = real_img<threshold
    fake_img_mask = fake_img<threshold
    # only leave black pixels 
    real_img = (1-real_img) * real_img_mask
    fake_img = (1-fake_img) * fake_img_mask
    
    # count pixels used for metric
    total = torch.count_nonzero(real_img_mask+fake_img_mask)
    # calculate pixelwise distance
    p_distance = torch.abs(real_img-fake_img).sum()
    
    return 1-(p_distance/total)

    