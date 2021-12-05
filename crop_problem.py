from pdfminer.layout import LAParams, LTTextBox,LTTextLine
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdf2image import convert_from_path

import re
import numpy as np
import cv2

import time
import os
import os.path as osp
from tqdm import tqdm

import argparse

def crop_problem(pdf_path: str) : 
    
    def scale(x,y,ori_h,ori_w,h,w) :
        return int(x * ori_w / w) , ori_h - int(y * ori_h / h)
    
    os.mkdir('background_with_check')
    os.mkdir('background_with_check/with_check')
    os.mkdir('background_with_check/background')

        
    
    pdf_files = os.listdir(pdf_path)
    for pdf_file in tqdm(pdf_files) :
        images = convert_from_path(osp.join(pdf_path,pdf_file))
        img_w,img_h = images[0].size

    

        fp = open(osp.join(pdf_path,pdf_file),'rb')
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        pages = PDFPage.get_pages(fp)

        cnt = 1
        pad = 30
        LT,RB = [[] for _ in range(len(images))],[[] for _ in range(len(images))]
        nums  = [[] for _ in range(len(images))]

        numQ = ""
        for idx, page in enumerate(pages):
            interpreter.process_page(page)
            layout = device.get_result()
            for lobjs in layout:
                w,h = layout.bbox[2], layout.bbox[3]
                if isinstance(lobjs, LTTextBox) :
                    for lobj in lobjs :
                        for obj in lobj :
                            # assert type(obj) == LTChar
                            text = obj.get_text()
                            if len(numQ) != 2 :
                                numQ += text
                            else :
                                numQ = numQ[1] + text

                            if '①' in text or '②' in text or '③' in text or '④' in text :
                                x1,y1 = scale(obj.bbox[0],obj.bbox[3],img_h,img_w,h,w)
                                x2,y2 = scale(obj.bbox[2],obj.bbox[1],img_h,img_w,h,w)
                                nums[idx].append([(x1+x2)//2,(y1+y2)//2])

                            elif '⑤' in text :
                                x1,y1 = scale(obj.bbox[0],obj.bbox[3],img_h,img_w,h,w)
                                x2,y2 = scale(obj.bbox[2],obj.bbox[1],img_h,img_w,h,w)
                                nums[idx].append([(x1+x2)//2,(y1+y2)//2])

                                x2 = 1160 if x2 < 1169 else 2150
                                RB[idx].append([x2,y2+500])
                            elif re.search('^\d+[.]',numQ) is not None :
                                x,y = scale(obj.bbox[0],obj.bbox[3],img_h,img_w,h,w)
                                LT[idx].append([x-pad,y-pad])

    crop_images = []
    crop_images_check = []
    check_files = sorted(os.listdir('check_img'))

    for page , image in enumerate(images) :

        np_img = np.array(image)

        for idx, ((x1,y1),(x2,y2)) in enumerate(zip(LT[page],RB[page])) :
            if x1 > x2 or y1 > y2 or x2 -x1 > 1169 :
                continue
            # save ori_background
            crop_images.append(np_img[y1:y2,x1:x2,:].copy())
            
            # save check_background ( random 1-5 ) 
            for _ in range(np.random.randint(1,5)) :
                file_idx = np.random.randint(0,len(check_files))
                ch_img = cv2.imread(osp.join('check_img',check_files[file_idx]))
                ch_img = cv2.resize(ch_img,(100,100))

                h,w = ch_img.shape[:2]
                ch = np.ones((h,w,3)).astype(np.uint8) * 255
                ch_mask = np.zeros((h,w,1)).astype(np.uint8)

                x,y,_ = np.where(ch_img < 200)

                for x_,y_ in zip(x,y) :
                    ch[x_,y_,:] = 0
                    ch_mask[x_,y_,:] = 255

                num = np.random.randint(5*idx,5*idx+5)
                nx,ny = nums[page][num]


                cv2.copyTo(ch,ch_mask,np_img[ny-50:ny+50,nx-50:nx+50,:])
            crop_images_check.append(np_img[y1:y2,x1:x2,:])

    for img,img_check in zip(crop_images,crop_images_check) :
        name = np.random.randint(int(1e6))
        
        cv2.imwrite(f'background_with_check/with_check/{name}.png',img_check[:,:,::-1])
        cv2.imwrite(f'background_with_check/background/{name}.png',img[:,:,::-1])

            
def ori_handwriting(hand_path: str) :
    hand_files = sorted(os.listdir(hand_path))

    os.makedirs('ori_hand',exist_ok=True)
    os.makedirs('ori_hand_mask',exist_ok=True)
    
    for hand_file in tqdm(hand_files) : 
        hand_img = cv2.imread(osp.join(hand_path,hand_file))

        h,w = hand_img.shape[:2]
        x,y, _ = np.where(hand_img < 100)

        hand = np.ones((h,w,3)).astype(np.uint8) * 255
        hand_mask = np.zeros((h,w,1)).astype(np.uint8)

        for x_,y_ in zip(x,y) :
            hand[x_,y_,:] = 125
            hand_mask[x_,y_,:] = 255

        if w > 500 : 
            hand = cv2.resize(hand,(500,int(h/w*500)))
            hand_mask = cv2.resize(hand_mask,(500,int(h/w*500)))

        cv2.imwrite(f"ori_hand/{hand_file}",hand)
        cv2.imwrite(f"ori_hand_mask/{hand_file}",hand_mask)
        
def make_img(save_dir:str) :
    img_files = sorted(os.listdir('background'))
    hand_files = sorted(os.listdir('ori_hand'))

    os.makedirs(save_dir,exist_ok=True)
    for img_file in tqdm(img_files) :
        img = cv2.imread(osp.join('background',img_file))

        idx = np.random.randint(0,len(hand_files))
        hand = cv2.imread(osp.join('ori_hand',hand_files[idx]))
        mask = cv2.imread(osp.join('ori_hand_mask',hand_files[idx]))

        dx,dy = np.random.randint(0,img.shape[1]-500), np.random.randint(0,img.shape[0]-hand.shape[0])
        cv2.copyTo(hand,mask,img[dy:dy+hand.shape[0],dx:dx+hand.shape[1]])
        cv2.imwrite(osp.join(save_dir,img_file),img)
        
    print(f"Save Image to {save_dir} with handwriting")
        
if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser(description="Make image with handwriting")
    parser.add_argument("--pdf_path", type=str, help="load pdf path")
    parser.add_argument("--hand_path", type=str, help="load hand path")
    parser.add_argument("--save_dir", type=str, help="save image with handwriting")
    
    args = parser.parse_args()
    
    crop_problem(args.pdf_path)
    ori_handwriting(args.hand_path)
    make_img(args.save_dir)
    