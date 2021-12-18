

# import streamlit as st

# html = """
#   <style>
#     /* Disable overlay (fullscreen mode) buttons */
#     .overlayBtn {
#       display: none;
#     }

#     /* Remove horizontal scroll */
#     .element-container {
#       width: auto !important;
#     }

#     .fullScreenFrame > div {
#       width: auto !important;
#     }

#     /* 2nd thumbnail */
#     .element-container:nth-child(4) {
#       top: -266px;
#       left: 350px;
#     }

#     /* 1st button */
#     .element-container:nth-child(3) {
#       left: 10px;
#       top: -60px;
#     }

#     /* 2nd button */
#     .element-container:nth-child(5) {
#       left: 360px;
#       top: -326px;
#     }
#   </style>
# """
# st.markdown(html, unsafe_allow_html=True)

# st.image("save/1_paper2_0.jpg", width=300)
# st.button("Show", key=1)

# st.image("save/1_paper2_1.jpg", width=300)
# st.button("Show", key=2)
# container = st.container()

# check = [False,True,True]

# for i in range(3):
#     check[i] = st.checkbox("Click")


# all = st.checkbox("Select all")
 
# if all:
#     selected_options = container.multiselect("Select one or more options:",
#          ['A', 'B', 'C'],['A', 'B', 'C'])
# else:
#     selected_options =  container.multiselect("Select one or more options:",
#         ['A', 'B', 'C'])

import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=150,
    drawing_mode=drawing_mode,
    initial_drawing = None,#{'version': '4.4.0', 'objects': [{'type': 'rect', 'fill': 'rgba(255,165,0,0.3)', 'stroke': '#000', 'left': 505, 'top': 106, 'width': 791, 'height': 218}, {'type': 'rect', 'fill': 'rgba(255,165,0,0.3)', 'stroke': '#000', 'left': 95, 'top': 94, 'width': 379, 'height': 194}], 'background': ''},
    key="canvas",
)

if canvas.json_data is not None:
    objects = pd.json_normalize(canvas.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)


canvas_result = st_canvas(initial_drawing = canvas.json_data)

if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)

# canvas_result = st_canvas(
#     fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#     stroke_width=stroke_width,
#     stroke_color=stroke_color,
#     background_color=bg_color,
#     background_image=Image.open(bg_image) if bg_image else None,
#     update_streamlit=realtime_update,
#     height=150,
#     drawing_mode='rect',
#     key="canvas",
# )

# # Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)
