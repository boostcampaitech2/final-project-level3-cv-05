from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import streamlit as st

def crop_canvas( canvas_result, img):
    """"
    input
    streamlit ui place,
    image PIL
    flag : for return all or one 
    """
    img = img.resize((900,1000))
    img = np.array(img)

    if "new_images" not in st.session_state:
        st.session_state["new_images"] = []
    for object in canvas_result.json_data["objects"]:
        x = object["left"]
        y = object["top"]
        w = object["width"]
        h = object["height"]

        new_img = img[y:y+h,x:x+w]
        new_img = cv2.resize(new_img,dsize=(0, 0),fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        st.session_state["new_images"].append(new_img)

    return st.session_state["new_images"]
