import os
import cv2
import numpy as np
import json 
from copy import deepcopy
from torch.utils.data import Dataset
import albumentations as A
import albumentations.pytorch

class CustomDataset(Dataset):
    def __init__(self, background_dir, handwriting_dir, check_dir, num_pos_json, mark_dir, background_transform=None, 
                 handwriting_transform=None, num_handwriting=15, segmentation=False, overlap=False):

        self.background_path = background_dir
        self.handwriting_path = handwriting_dir
        self.check_path = check_dir
        self.mark_path = mark_dir
        
        self.num_handwriting = num_handwriting
        self.background_transform = background_transform
        self.handwriting_transform = handwriting_transform
        self.segmentation = segmentation
        self.overlap = overlap

        with open(num_pos_json) as json_file :
            self.num_pos = json.load(json_file)
        
        self.background_images = os.listdir(self.background_path)
        self.handwriting_images = os.listdir(self.handwriting_path)
        self.check_images = os.listdir(self.check_path)
        self.mark_images = os.listdir(self.mark_path)
        
    def __len__(self):
        return len(self.background_images)
    
    def __getitem__(self, idx):
        ori_img = cv2.imread(os.path.join(self.background_path, self.background_images[idx]))
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        
        hand_img = deepcopy(ori_img)
        hand_img, seg_img = self.insert_handwriting_math(hand_img)
        hand_img, seg_img = self.insert_handwriting_mark(hand_img, seg_img)
        hand_img, seg_img = self.insert_handwriting_check(hand_img, seg_img, self.background_images[idx])

        if self.background_transform : 
            ori_img  = self.background_transform(image=ori_img)['image']
            hand_img = self.background_transform(image=hand_img)['image']

        if self.segmentation:
            if ori_img.shape[1:] != seg_img.shape[:2] :
                seg_img = cv2.resize(seg_img,(ori_img.shape[2],ori_img.shape[1]))
            return ori_img, hand_img, seg_img
        else:
            return ori_img, hand_img


    def insert_handwriting_math(self, background_img):
        seg_img = None
        if self.segmentation:
            seg_img = np.zeros(background_img.shape[:2]).astype(np.uint8)
            seg_img[background_img[:,:,0] < 200] = 1
        
        num_insert = 0
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
            
            # Draw handwriting math
            # TO DO : match background size
            random_size = np.random.randint(48, 96)
            handwriting_img = cv2.resize(handwriting_img, dsize=(random_size, random_size), interpolation=cv2.INTER_AREA)
            handwriting_img_mask = np.zeros((random_size,random_size,1)).astype(np.uint8)
            handwriting_img_mask[handwriting_img[:,:,0] != 255] = 255
            
            if self.handwriting_transform :
                trasnformed = self.handwriting_transform(image=handwriting_img,mask=handwriting_img_mask)
                handwriting_img = trasnformed['image']
                handwriting_img_mask = trasnformed['mask']
                
            for _ in range(100):    
                y1 = np.random.randint(background_img.shape[0])
                x1 = np.random.randint(background_img.shape[1])

                y2 = np.clip(y1 + handwriting_img.shape[0], 0, background_img.shape[0])
                x2 = np.clip(x1 + handwriting_img.shape[1], 0, background_img.shape[1])
                
                # wrong pos
                if y1 > y2 or x1 > x2 : 
                    continue
                
                # check overlap
                if not self.overlap and background_img[y1:y2, x1:x2][background_img[y1:y2, x1:x2] < background_threshold].size != 0 : 
                    continue 
                
                cv2.copyTo(handwriting_img,handwriting_img_mask,background_img[y1:y2, x1:x2])                        
                if self.segmentation:
                    cv2.copyTo(np.full(handwriting_img.shape[:2],2).astype(np.uint8),handwriting_img_mask,seg_img[y1:y2, x1:x2])
                break

        return background_img, seg_img
    
    
    def insert_handwriting_check(self, background_img, seg_img, img_file_name):
        # Draw Check image
        rand_pos_idx = np.random.randint(5)
        nx,ny = self.num_pos[img_file_name][rand_pos_idx]
        
        rand_check_idx = np.random.randint(len(self.check_images))
        check_img = cv2.imread(os.path.join(self.check_path, self.check_images[rand_check_idx]))
        
        random_size = np.random.choice(range(50,100,2),1)[0]
        check_img = cv2.resize(check_img,(random_size,random_size))
        check_img_mask = np.zeros((random_size,random_size,1)).astype(np.uint8)
        check_img_mask[check_img[:,:,0] < 200] = 255
        
        if self.handwriting_transform :
            trasnformed = self.handwriting_transform(image=check_img,mask=check_img_mask)
            check_img = trasnformed['image']
            check_img_mask = trasnformed['mask']
                
        pad = random_size // 2
        cv2.copyTo(check_img,check_img_mask,background_img[ny-pad:ny+pad,nx-pad:nx+pad])
        if self.segmentation:
            cv2.copyTo(np.full(check_img.shape[:2],3).astype(np.uint8),check_img_mask,seg_img[ny-pad:ny+pad,nx-pad:nx+pad])
                       
        return background_img, seg_img
    
    def insert_handwriting_mark(self, background_img, seg_img):
        # Draw Mark image
        rand_mark_idx = np.random.randint(len(self.mark_images))
        mark_img = cv2.imread(os.path.join(self.mark_path, self.mark_images[rand_mark_idx]))
        mark_img = cv2.cvtColor(mark_img, cv2.COLOR_BGR2RGB)
        
        random_size = np.random.choice(range(50,90,2),1)[0]
        mark_img = cv2.resize(mark_img,(random_size,random_size))
        mark_img_mask = np.zeros((random_size,random_size,1)).astype(np.uint8)
        mark_img_mask[mark_img[:,:,2] < 200] = 255
        
        if self.handwriting_transform :
            trasnformed = self.handwriting_transform(image=mark_img,mask=mark_img_mask)
            mark_img = trasnformed['image']
            mark_img_mask = trasnformed['mask']

        cv2.copyTo(mark_img,mark_img_mask,background_img[20:20+random_size,:random_size])
        if self.segmentation:
            cv2.copyTo(np.full(mark_img.shape[:2],4).astype(np.uint8),mark_img_mask,seg_img[20:20+random_size,:random_size])
                       
        return background_img, seg_img
    
        
    
if __name__ == "__main__":
    background_dir = 'background'
    handwriting_dir = 'ori_hand'
    check_dir = 'check_img'
    num_pos_json = 'pos.json'
    mark_dir = 'mark_img'
    # background_transform = A.Compose([
    #     A.Resize(512,512,cv2.INTER_AREA),
    #     A.RandomCrop(512,512),
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    #     A.pytorch.transforms.ToTensorV2()
    # ])
    background_transform = None
    handwriting_transform = None
    
    np.random.seed(42)
    dataset = CustomDataset(background_dir, handwriting_dir, check_dir, num_pos_json, 
                            mark_dir, num_handwriting = 5, 
                            background_transform = background_transform,
                            handwriting_transform = handwriting_transform,
                            segmentation=True)    
    ori_img, hand_img, seg_img = dataset[0]
    cv2.imwrite(f'loader_test/ori_img1.jpg', np.array(ori_img.permute(1,2,0)))
    cv2.imwrite(f'loader_test/hand_img1.jpg', np.array(hand_img.permute(1,2,0)))
    cv2.imwrite(f'loader_test/seg_img1.png', np.array(seg_img))

    dataset = CustomDataset(background_dir, handwriting_dir, check_dir, num_pos_json, num_handwriting = 5, 
                            background_transform = background_transform,
                            handwriting_transform = handwriting_transform,
                            segmentation=False)    
    ori_img, hand_img = dataset[0]
    cv2.imwrite(f'loader_test/ori_img2.jpg', np.array(ori_img.permute(1,2,0)))
    cv2.imwrite(f'loader_test/hand_img2.jpg', np.array(hand_img.permute(1,2,0)))

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