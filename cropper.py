#pip install streamlit-cropper 
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import os
import cv2


def crop(img_file):
    st.set_option('deprecation.showfileUploaderEncoding', False)

    #img_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])

    if img_file:
        img = Image.open(img_file)
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF',
                                    aspect_ratio=None)
        
        # Manipulate cropped image at will
        st.write("Preview")
        #_ = cropped_img.thumbnail((200,300))
        st.image(cropped_img)

        save = st.button("SAVE")
        if save:
            #TO DO : Answer , problem id, problem name
            #SAVE Image
            if os.path.isdir("save") == False:
                os.mkdir("save")

            cropped_img.save("save/{}".format(img_file.name))




if __name__ == '__main__':
    crop()