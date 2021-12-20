import streamlit as st
from PIL import Image
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
    return img 

def upload_problem_images(place, router):
    #TO DO : Get New Data, then reset cache
    image_file = place.file_uploader("Upload Image",type=['png','jpeg','jpg'])
    img = None 

    if image_file is not None:
        img = load_image(image_file) #Get Image
        img = img.convert('RGB') #RGBA -> RGB
        img = img.resize((1000,900))
        st.session_state["image"] = img #Give Image
         #TO DO : Find Error cuase over 50MB sol:Resize, change to ratio (1080x1920)
        place.image(img)
        _, _, next = place.columns(3)

        if next.button("다음"):
            st.session_state["file_name"] = image_file.name
            st.session_state["sub_page"] = "second"
            page_chg('/',router)

def run_object_detection(img, place, router):
    if 'locations' not in st.session_state:
        _, _, st.session_state['locations'] = OD_image(img)
    if 'json_file' not in st.session_state:
        st.session_state['json_file'] = crop_editor.make_detection_canvas(st.session_state['locations'])

    #Get locations only once
    if st.session_state['json_file'] is not None:
        # Specify canvas parameters in application
        place.write("Load가 끝날 때마다 천천히 조정해야 수정이 적용됨.")
        select = place.selectbox("Tool:", ("크기 조정", "새로 그리기"))
        drawing_mode = {"크기 조정":"transform","새로 그리기":"rect"}
        bg_image = img
        shape = np.shape(bg_image)
        place.write("다 했으면, 아래 canvas 아래에서 저장 버튼을 눌러주세요.")

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width= 1,
            stroke_color= "#000",
            background_color= "#eee",
            background_image= bg_image,
            update_streamlit= False, #real time update
            height= shape[0], # To Do :  set ratio
            width = shape[1],
            drawing_mode=drawing_mode[select],
            #이동은 되는데, 크기 조정이 안됨. 
            #json file이 아닌 image file로하기.
            initial_drawing =  st.session_state['json_file'],
            key = "canvas"
        )

        _, save, next = place.columns(3)
        images = place.empty()
        flag_a = next.button("다음")
        if 'crop_image' not in st.session_state:
            st.session_state["crop_image"] = None

        if save.button("SAVE"):
            cropped_imges = []
            if canvas_result.json_data["objects"] is None:
                place.error("자를 문제가 없습니다.")
            else:
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

                for cnt in contours:
                    x,y,w,h=cnt
                    if (x,x+w,y,y+h) == (0,1000,0,900):
                        #print("yes")
                        continue
                    area = (x,y,x+w,y+h)
                    #print(area)
                    cropped_img = img.crop(area)
                    cropped_imges.append(np.array(cropped_img))
                st.session_state["crop_image"] = cropped_imges
                
                place.write("자른 문제 결과")
                if st.session_state["crop_image"] is not None:
                    place.image(cropped_imges)

        if flag_a and (st.session_state["crop_image"] is not None):
            st.session_state['sub_page'] = "third"
            page_chg('/',router)
        elif flag_a: #only push button
            place.error("자른 문제를 저장해주세요")

@st.cache
def seg_image(image):
    pass
    return image

def run_gan(place, router):
    crop_images = st.session_state["crop_image"]
    gan_images = None

    place.subheader("손글씨 지운 사진 확인하고, 문제의 과목과 답을 입력하세요.") #Show Clear image
    place.subheader("문제의 과목과 답을 모두 입력 후에, 문제들을 저장하세요.")
    #only run once
    #해당 페이지에서 다시 새로고침하면, 뜨지 않음.
    if "idx" not in st.session_state:
        st.session_state['idx'] = 0
        st.session_state['subject'] = dict()
        st.session_state['answer'] = dict()

    if gan_images is None:
        gan_images = GAN_image(crop_images)
    else:
        pass

    place.image(gan_images[st.session_state['idx']])
    #write subject and answer if saved
    if (st.session_state['idx'] in st.session_state['subject'].keys()) and (st.session_state["idx"] in st.session_state["answer"].keys()):
        place.write("과목 : {}, 정답 : {}".format(st.session_state['subject'][st.session_state['idx']], st.session_state['answer'][st.session_state['idx']]))
    else:
        place.warning("아직 저장되지 않은 문제입니다.")

    
    before_p, save,next_p = place.columns(3)

    if next_p.button("다음 문제"):
        if st.session_state['idx'] < len(gan_images)-1:
            st.session_state['idx'] += 1
        else:
            st.session_state['idx'] = len(gan_images)-1
            st.warning("It's last problem")
        page_chg('/',router)

    if before_p.button("이전 문제"):
        if st.session_state['idx'] > 0:
            st.session_state['idx'] -= 1
        else:
            st.session_state['idx'] = 0
            st.warning("It's first problem")
        page_chg('/',router)

    col1, col2, col3 = place.columns(3)
    sub = col1.text_input("과목은?")
    ans = col2.text_input("정답은?")

    #SAVE subject, answer
    if col3.button("과목,답 저장"):
        if (sub!='') and (ans!=''): 
            st.session_state['subject'][st.session_state['idx']] = sub
            st.session_state['answer'][st.session_state['idx']] = ans
            st.success("과목과 정답을 저장했습니다.")
        else:
            st.warning("과목과 정답을 모두 기재해주세요.")

    if save.button("마지막! 문제들 저장"):
        if os.path.isdir("save")==False:
            os.mkdir("save")
        
        #Check write all answer
        if (len(st.session_state['answer']) == len(gan_images)):
            for i in range(len(gan_images)):
                save_name = 'save/%s_%s_%d.jpg'%(st.session_state['user_id'], st.session_state["file_name"][:-4] , i)
                #save image to file save/
                cv2.imwrite(save_name,gan_images[i])
                # save img path in db
                query = 'insert into problems (user_id, problem_file_name, answer, subject) values ("%s", "%s", "%s","%s");'%(st.session_state['user_id'], save_name, st.session_state['answer'][i],st.session_state['subject'][i])
                rowcount = run_insert(query)
                if rowcount!=0:
                    st.info('문제가 저장되었습니다.')
        else:
            place.error("과목과 정답을 기재하지 않은 문제가 있습니다.")

    _, _, next = place.columns(3)
    if next.button("PDF 만들기"):
        del st.session_state['idx']
        st.session_state['sub_page'] = "fourth"
        page_chg('/',router)

def make_problem_pdf(place, router, images):
    #Choose Problem
    
    img1, img2, img3 = place.columns(3)
    ch1, ch2, ch3 = place.columns(3)

    if "idx_p" not in st.session_state:
        st.session_state["idx_p"] = 0
        st.session_state["pick_problem"] = []
    
    #Show images and get check
    try:
        img = images[st.session_state["idx_p"]]
        img1.image(Image.open(img[2]))
        if ch1.button("1"):
            st.session_state["pick_problem"].append(st.session_state["idx_p"])
    except IndexError:
        img1.empty()
    
    try:
        img = images[st.session_state["idx_p"] + 1]
        img2.image(Image.open(img[2]))
        if ch2.button("2"):
            st.session_state["pick_problem"].append(st.session_state["idx_p"]+1)
    except IndexError:
        img2.empty()
    
    try:
        img = images[st.session_state["idx_p"] + 2]
        img3.image(Image.open(img[2]))
        if ch3.button("3"):
            st.session_state["pick_problem"].append(st.session_state["idx_p"]+2)
    except IndexError:
        img3.empty()

    #Next button
    if place.button("다음으로"):
        if len(images) > st.session_state["idx_p"]:
            st.session_state["idx_p"] += 3
            page_chg('/',router)
        else:
            st.warning("문제가 끝났습니다.")
    
    #print(st.session_state["pick_problem"])

    export_as_problem_pdf = st.button("Export Problem Report")
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
    
    export_as_answer_pdf = st.button("Export Answer Report")
    if export_as_answer_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        for q_n, i in enumerate(st.session_state["pick_problem"]):
            text = "N{} : A {} \n".format(str(q_n+1).zfill(2),images[i][3])
            pdf.multi_cell(40,10,text)
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
        st.markdown(html, unsafe_allow_html = True)
    
    _, col2 = place.columns(2)
    if col2.button("처음으로"):
        st.session_state['sub_page'] = "first"
        page_chg('/',router)