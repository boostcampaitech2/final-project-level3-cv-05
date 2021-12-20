import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from utils.utils import *

from gan import GAN_image
from detection import OD_image
import numpy as np

import cv2
import crop_editor
import os
import base64

from fpdf import FPDF
import imagesize

#pdf 다운
def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    return img 

def upload_problem_images(place, router):
    #TO DO : Get New Data, then reset cache
    image_file = place.file_uploader("Upload Image",type=['png','jpeg','jpg'])
    img = None 

    if image_file is not None:
        img = load_image(image_file) #Get Image
        img = img.convert('RGB') #RGBA -> RGB
        st.session_state["image"] = img
         #TO DO : Find Error cuase over 50MB sol:Resize, change to ratio (1080x1920)
        _, _, next = place.columns(3)
        place.image(st.session_state["image"])

        if next.button("다음"):
            st.session_state["file_name"] = image_file.name
            st.session_state["sub_page"] = "second"
            page_chg('/',router)

def run_object_detection(img, place, router):
    place.subheader("틀린 문제를 잘랐습니다. 한 장씩 확인하며 수정할 부분은 수정하고 저장해주세요.") 
    #only run once
    #해당 페이지에서 다시 새로고침하면, 뜨지 않음.
    if "idx" not in st.session_state:
        st.session_state['idx'] = 0
        _, st.session_state["crop_images"], _ = OD_image(img)

    if "crop_images" in st.session_state:
        try:
            place.image(st.session_state["crop_images"][st.session_state['idx']])
        except IndexError: #detection 결과가 없을 때
            place.write("틀린 문제를 찾지 못했습니다. 사용자가 직접 문제들을 그려주세요")

            canvas_result = st_canvas(
            fill_color = "rgba(255,165,0,0.3)",
            stroke_width = 1,
            stroke_color = "#000",
            background_color = "#eee",
            background_image = img.resize((1000,900)),
            update_streamlit = True,
            height = 1000,
            width = 900,
            drawing_mode  = "rect",
            key = "canvas"
            )

            if place.button("그린 문제 저장"):
                if canvas_result.json_data is None:
                    st.warning("문제 위에 박스를 치세요")
                else:
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
                            new_images.append(new_img)
                        
                    st.info("문제들이 저장되었습니다.")
                    st.session_state["crop_images"] = new_images
        else:
            canvas, next_p, next = place.columns(3)

            if next_p.button("다음 문제"):
                if st.session_state['idx'] < len(st.session_state["crop_images"])-1:
                    st.session_state['idx'] += 1
                else:
                    st.session_state['idx'] = len(st.session_state["crop_images"])-1
                    st.warning("마지막 문제 입니다.")
                page_chg('/',router)

            #SAVE Image or draw new box : 틀린 문제 말고도 사용자가 더 원하는 문제가 있으면 자를 수 있도록 하면 좋을 듯
            if canvas.checkbox("새로 그리기"):
                #새로 그리기 누르면 canvas 나타나고
                #거기서 나온 crop image 로 교체
                canvas_result = st_canvas(
                    fill_color = "rgba(255,165,0,0.3)",
                    stroke_width = 1,
                    stroke_color = "#000",
                    background_color = "#eee",
                    background_image = img.resize((1000,900)),
                    update_streamlit = True,
                    height = 1000,
                    width = 900,
                    drawing_mode  = "rect",
                    key = "canvas"
                )

                print("yes")
                if place.button("그린 문제로 저장"):
                    print("hi")
                    if canvas_result.json_data is None:
                        st.warning("문제 위에 박스를 치세요")
                    else:
                        object = canvas_result.json_data["objects"][0]
                        x = object["left"]
                        y = object["top"]
                        w = object["width"]
                        h = object["height"]

                        if not isinstance(img, np.ndarray):
                            img = img.resize((900,1000))
                            img = np.array(img)
                        new_img = img[y:y+h,x:x+w]
                        place.image(new_img)

                        st.session_state["crop_images"][st.session_state["idx"]] = new_img
                        st.info('문제가 새로 저장되었습니다.')

            if next.button("글씨 지우기"):
                del st.session_state["idx"]
                st.session_state['sub_page'] = "third"
                page_chg('/',router)

@st.cache
def run_seg(image):
    pass
    return image

def run_gan(place, router):


    place.subheader("손글씨 지운 사진 확인하고, 문제의 과목과 답을 입력하세요.") #Show Clear image
    place.subheader("문제의 과목과 답을 입력 후에, 문제들을 저장하세요.")
    #only run once
    #해당 페이지에서 다시 새로고침하면, 뜨지 않음.
    if "idx" not in st.session_state:
        st.session_state['idx'] = 0
        st.session_state["gan_images"] = GAN_image(st.session_state["crop_images"])
        del st.session_state["crop_images"]

    if "gan_images" in st.session_state:
        place.image(st.session_state["gan_images"][st.session_state['idx']])
    before_p, save, next_p = place.columns(3)
    sub = place.text_input("과목은?")
    ans = place.text_input("정답은?")

    if next_p.button("다음 문제"):
        if st.session_state['idx'] < len(st.session_state["gan_images"])-1:
            st.session_state['idx'] += 1
        else:
            st.session_state['idx'] = len(st.session_state["gan_images"])-1
            st.warning("마지막 문제 입니다.")
        page_chg('/',router)

    if before_p.button("이전 문제"):
        if st.session_state['idx'] > 0:
            st.session_state['idx'] -= 1
        else:
            st.session_state['idx'] = 0
            st.warning("첫 번째 문제 입니다.")
        page_chg('/',router)

    #SAVE subject, answer, image
    if save.button("저장"):
        if os.path.isdir("save")==False:
            os.mkdir("save")
        
        if (sub!='') and (ans!=''):
            save_name = 'save/%s_%s_%d.jpg'%(st.session_state['user_id'], st.session_state["file_name"][:-4] ,st.session_state["idx"]) 
            #save image to file save/
            cv2.imwrite(save_name,st.session_state["gan_images"][st.session_state['idx']])
            query = 'insert into problems (user_id, problem_file_name, answer, subject) values ("%s", "%s", "%s","%s");'%(st.session_state['user_id'], save_name, ans,sub)
            rowcount = run_insert(query)
            if rowcount!=0:
                    st.info('문제가 저장되었습니다.')
        else:
            st.warning("과목과 정답을 모두 기재해주세요.")

    _, _, next = place.columns(3)
    if next.button("PDF 만들기"):
        del st.session_state['idx']
        del st.session_state["gan_images"]
        st.session_state['sub_page'] = "fourth"
        page_chg('/',router)

def make_problem_pdf(place, router, images, flag):
    #Choose Problem
    
    img1, img2, img3 = place.columns(3)
    ch1, ch2, ch3 = place.columns(3)

    if "idx_p" not in st.session_state:
        st.session_state["idx_p"] = 0
        st.session_state["pick_problem"] = []
    
    #Show images and get check
    try:
        img = images[st.session_state["idx_p"]]
        img1.image(load_image(img[2]))
        if ch1.button("{}".format(st.session_state["idx_p"]+1)):
            st.session_state["pick_problem"].append(st.session_state["idx_p"])
            place.success("{} 번 문제 저장".format(st.session_state["idx_p"]+1))
    except IndexError:
        img1.empty()
    
    try:
        img = images[st.session_state["idx_p"] + 1]
        img2.image(load_image(img[2]))
        if ch2.button("{}".format(st.session_state["idx_p"]+2)):
            st.session_state["pick_problem"].append(st.session_state["idx_p"]+1)
            place.success("{} 번 문제 저장".format(st.session_state["idx_p"]+2))
    except IndexError:
        img2.empty()
    
    try:
        img = images[st.session_state["idx_p"] + 2]
        img3.image(load_image(img[2]))
        if ch3.button("{}".format(st.session_state["idx_p"]+3)):
            st.session_state["pick_problem"].append(st.session_state["idx_p"]+2)
            place.success("{} 번 문제 저장".format(st.session_state["idx_p"]+3))
    except IndexError:
        img3.empty()

    #Next button
    if place.button("다음으로"):
        if len(images) > st.session_state["idx_p"]:
            st.session_state["idx_p"] += 3
            page_chg('/',router)
        else:
            st.warning("문제가 끝났습니다.")

    export_as_problem_pdf = st.button("문제지 출력")
    if export_as_problem_pdf:
        if os.path.isdir("save"):
            # A4 size [width : 210, height : 287]
            pdf = FPDF()
            pdf.set_font('Arial', 'B', 24)
            pdf.set_text_color(0, 0, 0)

            x, y, w = 10, 20, 90
            problem_pad = 30
            pdf.add_page()
            # 단 나누기
            pdf.line(105, 10, 105, 280)
            for q_n, i in enumerate(st.session_state["pick_problem"]):
                img = images[i][2]
                img_w, img_h = imagesize.get(img)
                h = int(w * (img_h/img_w))

                if y+h+problem_pad>=287:
                    # 단 이동
                    if x==10:
                        x=110
                        y=20
                    # 페이지 이동
                    elif x==110:
                        pdf.add_page()
                        # 단 나누기
                        pdf.line(105, 10, 105, 280)
                        x=10
                        y=20
                    else:
                        raise

                # 문제 번호 작성
                pdf.text(x=x, y=y-3, txt='N'+str(q_n+1).zfill(2))
                # 문제 붙이기
                pdf.image(f"{img}", x=x, y=y, w=w, h=h)
                y = y+h+problem_pad
            html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
            st.markdown(html, unsafe_allow_html = True)
    
    export_as_answer_pdf = st.button("답지 출력")
    if export_as_answer_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        for q_n, i in enumerate(st.session_state["pick_problem"]):
            text = "N{} : A {} \n".format(str(q_n+1).zfill(2),images[i][3])
            pdf.multi_cell(40,10,text)
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
        st.markdown(html, unsafe_allow_html = True)
    
    if flag:
        _, col2 = place.columns(2)
        if col2.button("처음으로"):
            del st.session_state["idx_p"]
            st.session_state['sub_page'] = "first"
            page_chg('/',router)

