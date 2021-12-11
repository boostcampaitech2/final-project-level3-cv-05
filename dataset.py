import os
import cv2
import numpy as np
import json 
from copy import deepcopy
from torch.utils.data import Dataset
import albumentations as A


class CustomDataset(Dataset):
    def __init__(self, background_dir, handwriting_dir, check_dir, num_pos_json, background_transform=None, 
                 handwriting_transform=None, num_handwriting=15, segmentation=False):

        self.background_path = background_dir
        self.handwriting_path = handwriting_dir
        self.check_path = check_dir
        
        self.num_handwriting = num_handwriting
        self.background_transform = background_transform
        self.handwriting_transform = handwriting_transform
        self.segmentation = segmentation
        
        with open(num_pos_json) as json_file :
            self.num_pos = json.load(json_file)
        
        self.background_images = os.listdir(self.background_path)
        self.handwriting_images = os.listdir(self.handwriting_path)
        self.check_images = os.listdir(self.check_path)
        
    def __len__(self):
        return len(self.background_images)
    
    def __getitem__(self, idx):
        ori_img = cv2.imread(os.path.join(self.background_path, self.background_images[idx]))
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        hand_img = deepcopy(ori_img)
        
        hand_img, seg_img = self.insert_handwriting(hand_img, self.background_images[idx])

        
        if self.segmentation:
            return ori_img, hand_img, seg_img
        else:
            return ori_img, hand_img


    def insert_handwriting(self, background_img, img_file_name):
        seg_img = None
        if self.segmentation:
            seg_img = np.zeros(background_img.shape[:2])
            seg_img[background_img[:,:,0] != 255] = 1
        num_insert = 0
        
        if self.background_transform : 
            background_img = self.background_transform(image=background_img)['image']
        
        while num_insert < self.num_handwriting:
            num_insert += 1
            rand_idx = np.random.randint(len(self.handwriting_images))
            handwriting_img = cv2.imread(os.path.join(self.handwriting_path, self.handwriting_images[rand_idx]))
            handwriting_img = cv2.cvtColor(handwriting_img, cv2.COLOR_BGR2GRAY)

            background_threshold = 128
            shape = handwriting_img.shape
            white_background = len(handwriting_img[handwriting_img < background_threshold]) / shape[0] / shape[1]

            if white_background > 0.2:
                num_insert -= 1
                continue
            handwriting_img[handwriting_img > background_threshold] = 255

            handwriting_img = cv2.cvtColor(handwriting_img, cv2.COLOR_GRAY2RGB)
            # random_size = np.random.randint(64, 128)
            random_size = 128
            handwriting_img = cv2.resize(handwriting_img, dsize=(random_size, random_size), interpolation=cv2.INTER_AREA)
            handwriting_img_mask = np.zeros((random_size,random_size,1)).astype(np.uint8)
            
            x,y,_ = np.where(handwriting_img != 255)
            for x_,y_ in zip(x,y) :
                handwriting_img_mask[x_,y_,:] = 255
            # handwriting_img_mask[handwriting_img != 255] = 255
            
            if self.handwriting_transform :
                trasnformed = self.handwriting_transform(image=handwriting_img,mask=handwriting_img_mask)
                handwriting_img = trasnformed['image']
                handwriting_img_mask = trasnformed['mask']
            for _ in range(100):    
                y1 = np.random.randint(background_img.shape[0])
                x1 = np.random.randint(background_img.shape[1])

                y2 = np.clip(y1 + handwriting_img.shape[0], 0, background_img.shape[0])
                x2 = np.clip(x1 + handwriting_img.shape[1], 0, background_img.shape[1])
                
                exists = 0 < y2-y1 and 0 < x2-x1
                if exists and background_img[y1:y2, x1:x2][background_img[y1:y2, x1:x2] < background_threshold].size == 0:
                # if exists:
                    # background_img[y1:y2, x1:x2][handwriting_img[:y2-y1,:x2-x1] != 255] = handwriting_img[:y2-y1,:x2-x1][handwriting_img[:y2-y1,:x2-x1] != 255]
                    cv2.copyTo(handwriting_img,handwriting_img_mask,background_img[y1:y2, x1:x2])
                    if self.segmentation:
                        seg_img[y1:y2, x1:x2][handwriting_img[:y2-y1,:x2-x1,0] != 255] = 2
                    break
            
        # Draw Check image
        rand_check_idx = np.random.randint(len(self.check_images))
        check_img = cv2.imread(os.path.join(self.check_path, self.check_images[rand_check_idx]))

        # proper size 76
        check_img = cv2.resize(check_img,(76,76))
        check_img_mask = np.zeros((76,76,1)).astype(np.uint8)
        
        x,y,_ = np.where(check_img < 200)
        for x_,y_ in zip(x,y) :
            check_img_mask[x_,y_,:] = 255
        
        
        if self.handwriting_transform :
            trasnformed = self.handwriting_transform(image=check_img,mask=check_img_mask)
            check_img = trasnformed['image']
            check_img_mask = trasnformed['mask']
                
        
        rand_pos_idx = np.random.randint(5)
        nx,ny = self.num_pos[img_file_name][rand_pos_idx]

        # pad = size // 2
        cv2.copyTo(check_img,check_img_mask,background_img[ny-38:ny+38,nx-38:nx+38])
        # background_img[ny-38:ny+38,nx-38:nx+38][check_img < 200] = check_img[check_img < 200]

        return background_img, seg_img


if __name__ == "__main__":
    background_dir = '/opt/ml/final-project-level3-cv-05/background'
    handwriting_dir = '/opt/ml/final-project-level3-cv-05/handwriting'
    check_dir = '/opt/ml/final-project-level3-cv-05/check_img'
    num_pos_json = '/opt/ml/final-project-level3-cv-05/pos.json'

    np.random.seed(42)

    dataset = CustomDataset(background_dir, handwriting_dir, check_dir, num_pos_json, segmentation=True)
    ori_img, hand_img, seg_img = dataset[0]
    cv2.imwrite(f'/opt/ml/final-project-level3-cv-05/ori_img1.jpg', ori_img)
    cv2.imwrite(f'/opt/ml/final-project-level3-cv-05/hand_img1.jpg', hand_img)
    cv2.imwrite(f'/opt/ml/final-project-level3-cv-05/seg_img1.png', seg_img)

    dataset = CustomDataset(background_dir, handwriting_dir, check_dir, num_pos_json, segmentation=False)
    ori_img, hand_img = dataset[0]
    cv2.imwrite(f'/opt/ml/final-project-level3-cv-05/ori_img2.jpg', ori_img)
    cv2.imwrite(f'/opt/ml/final-project-level3-cv-05/hand_img2.jpg', hand_img)

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        drop_last=False
    )

    for idx, data_batch in enumerate(tqdm(data_loader)):
        pass