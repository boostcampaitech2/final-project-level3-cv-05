
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

    object = {'type': 'rect', 'version': '4.4.0', 'originX': 'left', 'originY': 'top',
    'left': 0, 'top': 0, 'width': 0, 'height': 0,
    'fill': 'rgba(255, 165, 0, 0.3)', 'stroke': '#000', 'strokeWidth': 1, 'strokeDashArray': None,
    'strokeLineCap': 'butt', 'strokeDashOffset': 0, 'strokeLineJoin': 'miter', 'strokeUniform': True,
    'strokeMiterLimit': 4, 'scaleX': 1, 'scaleY': 1, 'angle': 0, 'flipX': False, 'flipY': False,
    'opacity': 1, 'shadow': None, 'visible': True,
    'backgroundColor': '', 'fillRule': 'nonzero', 'paintFirst': 'fill', 'globalCompositeOperation': 'source-over',
    'skewX': 0, 'skewY': 0, 'rx': 0, 'ry': 0}

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
    


def crop_editor():
    #Get Detection point (x,y,width,height)
    points = [(0,274,152,93),(305,286,110,116)]
    json_file = make_detection_canvas(points)


    #Detection 부분 미세 조정 (transform)
    #Detection 못해도 사용자가 추가 가능 (rect)
    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("rect", "transform"))
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width= 1,
        stroke_color= "#000",
        background_color= "#eee",
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit= True,
        height= 500,
        width = 500,
        drawing_mode=drawing_mode,
        initial_drawing = json_file,
        key = "canvas"
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)
    
    
    if st.button("Save dataframe"):
        open('data.json','w').write(json.dumps(canvas_result.json_data, indent=4))


if __name__ == "__main__":
    crop_editor()