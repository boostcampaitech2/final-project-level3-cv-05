import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, background_dir, handwriting_dir, num_handwriting=15, transform=None):
        self.background_path = background_dir
        self.handwriting_path = handwriting_dir
        self.num_handwriting = num_handwriting
        self.transform = transform

        self.background_images = os.listdir(self.background_path)
        self.handwriting_images = os.listdir(self.handwriting_path)

    def set_transform(self, transform):
        self.transform = transform
    
    def __len__(self):
        return len(self.background_images)
    
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.background_path, self.background_images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        seg_img = np.zeros(img.shape[:2])
        seg_img[img[:,:,0] != 255] = 1

        img, seg_img = self.insert_handwriting(img, seg_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img, seg_img

    def insert_handwriting(self, background_img, seg_img):
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
            random_size = np.random.randint(64, 128)
            handwriting_img = cv2.resize(handwriting_img, dsize=(random_size, random_size), interpolation=cv2.INTER_AREA)

            for _ in range(100):    
                y1 = np.random.randint(background_img.shape[0])
                x1 = np.random.randint(background_img.shape[1])

                y2 = np.clip(y1 + handwriting_img.shape[0], 0, background_img.shape[0])
                x2 = np.clip(x1 + handwriting_img.shape[1], 0, background_img.shape[1])
                
                exists = 0 < y2-y1 and 0 < x2-x1
                if exists and background_img[y1:y2, x1:x2][background_img[y1:y2, x1:x2] < background_threshold].size == 0:
                    background_img[y1:y2, x1:x2][handwriting_img[:y2-y1,:x2-x1] != 255] = handwriting_img[:y2-y1,:x2-x1][handwriting_img[:y2-y1,:x2-x1] != 255]
                    seg_img[y1:y2, x1:x2][handwriting_img[:y2-y1,:x2-x1,0] != 255] = 2
                    break

        return background_img, seg_img


if __name__ == "__main__":
    background_dir = '/opt/ml/final-project-level3-cv-05/background'
    handwriting_dir = '/opt/ml/final-project-level3-cv-05/handwriting'
    
    dataset = CustomDataset(background_dir, handwriting_dir)
    img, seg_img = dataset[0]
    cv2.imwrite(f'/opt/ml/final-project-level3-cv-05/img.jpg', img)
    cv2.imwrite(f'/opt/ml/final-project-level3-cv-05/seg_img.png', seg_img)

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