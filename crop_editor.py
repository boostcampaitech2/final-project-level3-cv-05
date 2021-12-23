from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import streamlit as st
import copy

def crop_canvas( canvas_result, img):
    #Crop Image masking
    mask = canvas_result.image_data
    img_float32 = mask.astype("uint8")
    new = cv2.cvtColor(img_float32,cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(new,cv2.COLOR_BGR2GRAY) 
    ret2,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel=np.ones((3,3),np.uint8)
    dilated=cv2.dilate(mask,kernel,iterations=3)
    # dilated = dilated.astype("uint8")

    ### finding contours, can use connectedcomponents aswell
    contours,_ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours=[cv2.boundingRect(cnt) for cnt in contours]

    cropped_imges = []
    img = img.resize((900,1000))
    img = np.array(img)
    for cnt in contours:
        x,y,w,h=cnt
        print(x,x+w,y,y+h)
        if (x,x+w,y,y+h) == (0,900,0,1000):
            #print("yes")
            continue
        #print(area)
        cropped_img = img[y:y+h,x:x+w]
        cropped_img = cv2.resize(cropped_img,dsize=(0, 0),fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        cropped_imges.append(np.array(cropped_img))
    
    st.session_state["crop_images"] = cropped_imges

def change_point(img, points):
    w,h= img.size

    rx = 900/w
    ry = 1000/h

    object = {
        "type":"rect",
        "fill" : "rgba(255,165,0,0.3)",
        "stroke" : "#000"}

    objects_list = []
    if len(points)==0:
        return None
    else:
        for x1,y1,x2,y2 in points:
            object['left'] = rx * x1
            object['top'] = ry * y1
            object['width'] = (x2-x1)*rx
            object['height'] = (y2-y1)*ry
            
            copy_dict = copy.deepcopy(object)
            objects_list.append(copy_dict)
        
        json_file ={
            'version': '4.4.0',
            'objects': objects_list,
            'background': '#eee'
            }

        return json_file


 
