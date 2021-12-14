from PIL import Image
from fpdf import FPDF

import streamlit as st
import os
import pandas as pd
import csv
import base64

#streamlit run app.py --server.address=127.0.0.1
import crop_editor

from detection import load_model, get_crop_location, draw_from_crop_locations, crop_from_crop_locations
from models.AttentionRes_G import AttentionRes_G
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

import json

#streamlit run server.py --server.address=127.0.0.1
#이렇게 하면 브라우저가 Local로 띄워짐.

#wide
st.set_page_config(layout="wide")

# Fxn
@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img 

@st.cache
def load_attgan_model(model_file):
    net = AttentionRes_G(3, 3, 64, n_blocks=9)
    load_path = './checkpoints/'+model_file
    state_dict = torch.load(load_path, map_location=str('cuda'))
    net.load_state_dict(state_dict)
    return net

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    images = []
    image_numpys = input_image.data.cpu().float().numpy()
    for image_numpy in image_numpys:
        # image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        images.append(image_numpy.astype(imtype))
    return images


#Fxn to Save answer
def save_results(results_df,button_press,image_file,problem_name,answer):
    results_df.at[button_press,'File name'] = image_file.name
    results_df.at[button_press,'Nick name'] = problem_name
    results_df.at[button_press,'Answer'] = answer
    results_df.to_csv('answer.csv',index=None)


#Fxn to make csv file
def load_data():
    header = ["File name","Nick name","Answer"]
    try:
        df = pd.read_csv('answer.csv')
    except FileNotFoundError:
        with open('answer.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        df = pd.read_csv('answer.csv')
    return df


#Fxn to Save Upload csv
def save_uploaded_csv(uploadfile):
    if(os.path.isdir("math_data") == False): #Change path
        os.mkdir("math_data")

    with open(os.path.join("math_data",uploadfile.name),'wb') as f:
        f.write(uploadfile.getbuffer())
    return st.success("Save Answer : To Show Click Answer on Menu")


#Fxn to Save Uploaded File to Directory
def save_uploaded_file(uploadfile):
    if(os.path.isdir("math_data") == False): #Change path
        os.mkdir("math_data")

    with open(os.path.join("math_data",uploadfile.name),'wb') as f:
        f.write(uploadfile.getbuffer())
    return st.success("Upload file :{} in Server".format(uploadfile.name))


#Fxn to Save After File
def save_after_file(file,name):
    if(os.path.isdir("new_data")==False):
        os.mkdir("new_data")

    file.save('./new_data/{}'.format(name),'png')
    return st.success("Upload file :{} in Server".format(name))


#pdf 다운 (아직 완성 안됨)
def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


#Object Detection
@st.cache
def OD_image(image):
    ''' Crop uncorrect problem
    Parameters:
        image : PIL Image Type
    '''
    detector = load_model(cfg_path = "./checkpoints/yolov3_config.py", 
                    ckpt_path = "./checkpoints/yolov3_weight.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = detector.to(device)
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    locations = get_crop_location(detector, image)   # [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
    drawed_img = draw_from_crop_locations(image, locations)
    croped_img = crop_from_crop_locations(image, locations)
    return drawed_img, croped_img


#GAN
@st.cache
def GAN_image(images):
    ''' Erase Handwriting in image 
    Parameters:
        images : PIL Image Type
    '''
    # load model
    gan_model = load_attgan_model('attentiongan.pth')
    # data transform
    load_size = 512
    # img_transform = transforms.Compose(
    #     [transforms.Resize([load_size,load_size], Image.BICUBIC),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_transform = A.Compose(
        [A.Resize(load_size, load_size, 2),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensorV2()])
    inputs = []
    for image in images:
        inputs.append(img_transform(image=image)['image'])
    inputs = torch.stack(inputs, 0)
    # af_transform = img_transform(image)
    # c,w,h = af_transform.shape
    # af_transform = np.reshape(af_transform, (1,c,w,h)) # convert to batch form
    # forward and tensor to image
    with torch.no_grad():
        output = gan_model(inputs)
        outputs = tensor2im(output)
    return outputs


def streamlit_run():
    #result_df = load_data()
    button_press = 0

    st.title("Math wrong answer editor")

    menu = ["All","MakeImage","MakePDF","Answer","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "All":
        st.header("All part")
        st.subheader("1. Upload your problem images")

        #TO DO : Get New Data, then reset cache
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])

        if image_file is not None:
            img = load_image(image_file) #Get Image
            img = img.convert('RGB') #RGBA -> RGB
            st.image(img)
        
        st.subheader("2. Check wrong image, and you can edit")
        
        if 'OD_button' not in st.session_state:
            st.session_state.OD_button = False
        if st.button("Crop"):
            st.session_state.OD_button = True
        if st.session_state.OD_button:
            try:
                od_img, crop_images = OD_image(img)
            except UnboundLocalError:
                st.error("Plz, input image")
            else:
                #Show Image Crop
                st.subheader("Crop Result")
                st.image(od_img)
            #Use Crop Editor
            flag_edit = st.checkbox("Do you need to fix?")
            if flag_edit:
                crop_editor.crop_editor(image_file) 

                if(os.path.isfile("./data.json")):
                    crop_images = []
                    with open("data.json") as json_file:
                        json_data = json.load(json_file)
                        json_object = json_data["objects"]

                        for ob in json_object:
                            x = ob["left"]
                            y = ob["top"]
                            w = ob["width"]
                            h = ob["height"]

                            print(x,y,w,h)
                            area = (x,y,x+w,y+h)
                            #To Do : Ratio
                             
                            cropped_img = img.crop(area)
                            crop_images.append(cropped_img)

            if "OD_show_button" not in st.session_state:
                st.session_state.OD_show_button = False
            
            if st.button("Show"):
                st.session_state.OD_show_button = True
            
            if st.session_state.OD_show_button:
                st.image(crop_images) 

        st.subheader("3. Clear handwriting")

        if 'GAN_button' not in st.session_state:
            st.session_state.GAN_button = False
        if st.button("Clear"):
            st.session_state.GAN_button = True
        if st.session_state.GAN_button:
            gan_img = GAN_image(crop_images)

            st.subheader("Check Final Image & Save") #Show Clear image
            if "idx" not in st.session_state:
                st.session_state.idx = 0

            before, next, save = st.columns(3)

            if next.button("Next"):
                if st.session_state.idx <= len(gan_img)-2:
                    st.session_state.idx += 1
                else:
                    st.session_state.idx = len(gan_img)-1
                    st.warning("It's last problem")

            if before.button("Before"):
                if st.session_state.idx >= 1:
                    st.session_state.idx -= 1
                else:
                    st.session_state.idx = 0
                    st.warning("It's first problem")

            if save.button("Save"):
                if os.path.isdir("save")==False:
                    os.mkdir("save")
                
                for i in range(len(gan_img)):
                    cv2.imwrite('save/{}.jpg'.format(i),gan_img[i])
            
            st.image(gan_img[st.session_state.idx])

        
        st.subheader("4. Make Problem PDF")

        export_as_pdf = st.button("Export Report")

        if export_as_pdf:
            if os.path.isdir("save"):
                pdf = FPDF()

                w=120
                h=100

                for img in os.listdir("save"):
                    pdf.add_page()
                    pdf.image("save/"+img,x=0,y=10,w=w,h=h)

                pdf.set_font("Arial","B",16)

                html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")

                st.markdown(html, unsafe_allow_html = True)
            else:
                st.error("plz, save image")
    elif choice == "MakeImage":
        st.subheader("Upload your problem images")
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])

        if image_file is not None:
            #Get Before Image
            img = load_image(image_file)

            st.subheader("Before")
            st.image(img, use_column_width = True)

            od_img, crop_images = OD_image(img)
            gan_img = GAN_image(crop_images)
            # after_img = GAN_image(gan_img)

            flag_od = st.checkbox("Object Detection")
            flag_gan = st.checkbox("GAN")
            # flag_after = st.checkbox("AFTER")

            if flag_od:
                st.subheader("Object Detection")
                st.image(od_img,use_column_width = True)
            if flag_gan:
                st.subheader("GAN")
                st.image(gan_img[0],use_column_width = True)
            # if flag_after:
            #     st.subheader("AFTER")
            #     st.image(after_img,use_column_width = True)

            st.write(image_file.name)

            #saving file
            if st.button("Save"):
                save_uploaded_file(image_file)
                save_after_file(gan_img,image_file.name)
    elif choice == "Answer":
        st.text("Show")
    else:
        st.subheader("About")
        st.text("수학 오답 노트 편집기")
        st.text("친구구했조")
        st.text("Freinds")


if __name__ == '__main__':
    streamlit_run()