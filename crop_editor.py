from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import streamlit as st

def crop_canvas( canvas_result, img, place, flag):
    """"
    input
    streamlit ui place,
    image PIL
    flag : for return all or one 
    """
    new_images = []
    for object in canvas_result.json_data["objects"]:
        x = object["left"]
        y = object["top"]
        w = object["width"]
        h = object["height"]

        if not isinstance(img, np.ndarray):
            img = img.resize((900,1000))
            img = np.array(img)
            new_img = img[y:y+h,x:x+w]
            new_img = cv2.resize(new_img,dsize=(0, 0),fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            new_images.append(new_img)
        
        if flag: #detection 하지 못했을 때
            return new_images
        else:#한 문제의 이미지만 새로 만들때
            return new_images[0]