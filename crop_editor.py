
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
import copy

#pip install streamlit-drawable-canvas 
#설치해야함.

def make_detection_canvas(points):
    #(x,y,width,height)

    object = {
        "type":"rect",
        "fill" : "rgba(255,165,0,0.3)",
        "stroke" : "#000"}

    objects_list = []

    for x,y,width,height in points:
        object['left'] = x
        object['top'] = y
        object['width'] = width
        object['height'] = height

        copy_dict = copy.deepcopy(object)
        objects_list.append(copy_dict)

    json_file ={
        'version': '4.4.0',
        'objects': objects_list,
        'background': '#eee'
        }
    
    return json_file
    


def crop_editor(image):
    #Get Detection point (x,y,width,height)
    points = [(0,274,152,93),(305,286,110,116)]
    json_file = make_detection_canvas(points)
    #file open
    # with open("data.json","r") as f:
    #     json_file = json.load(f)
    

    #Detection 부분 미세 조정 (transform)
    #Detection 못해도 사용자가 추가 가능 (rect)
    # Specify canvas parameters in application
    drawing_mode = st.selectbox(
    "Drawing tool:", ("rect", "transform"))
    bg_image = image
    #bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width= 1,
        stroke_color= "#000",
        background_color= "#eee",
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit= True,
        height= 1000,
        width = 1000,
        drawing_mode=drawing_mode,
        #initial_drawing = json_file,
        key = "canvas"
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)

    object_list = []

    if canvas_result.json_data is not None:
        for object in canvas_result.json_data["objects"]:
            new_object = {
                "type":object["type"],
                "fill":object["fill"],
                "stroke":object["stroke"],
                "left":object["left"],
                "top":object["top"],
                "width":object["width"],
                "height":object["height"]}
            
            object_list.append(new_object)

        canvas_json = {
            "version" : "4.4.0",
            "objects" : object_list,
            "background" : "#eee"
        }
    
    if st.button("Save dataframe"):
        open('data.json','w').write(json.dumps(canvas_json, indent=4))


if __name__ == "__main__":
    crop_editor()