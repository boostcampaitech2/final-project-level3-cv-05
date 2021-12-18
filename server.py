from PIL import Image
from fpdf import FPDF

import streamlit as st
import extra_streamlit_components as stx
import os
import pandas as pd
import base64
from streamlit_drawable_canvas import st_canvas

import crop_editor
from gan import tensor2im

from detection import load_model, get_crop_location, draw_from_crop_locations, crop_from_crop_locations
from models.AttentionRes_G import AttentionRes_G

import albumentations as A
import torch
import cv2
import numpy as np
from utils.utils import *

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


#Fxn to Save Uploaded File to Directory
def save_uploaded_file(uploadfile):
    mkdir("math_data")
    with open(os.path.join("math_data",uploadfile.name),'wb') as f:
        f.write(uploadfile.getbuffer())
    return st.success("Upload file :{} in Server".format(uploadfile.name))


#Fxn to Save After File
def save_after_file(file,name):
    mkdir("new_data")  
    file.save('./new_data/{}'.format(name),'png')
    return st.success("Upload file :{} in Server".format(name))


#pdf 다운
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
    return drawed_img, croped_img, locations


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
    img_transform = A.Compose([
        A.Resize(load_size, load_size, 2),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        A.pytorch.ToTensorV2()])
    inputs = list()
    for image in images:
        inputs.append(img_transform(image=image)['image'])
    inputs = torch.stack(inputs, 0)
    with torch.no_grad():
        output = gan_model(inputs)
        outputs = tensor2im(output)
    return outputs

@st.cache
def seg_image(image):
    pass
    return image



def session_init():
    #session initialize
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    if 'auth_status' not in st.session_state:
        st.session_state['auth_status'] = None
    if 'after_join' not in st.session_state:
        st.session_state['after_join'] = None
    if 'prev_menu' not in st.session_state:
        st.session_state['prev_menu'] = ''


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
        #_, _, st.session_state['locations'] = OD_image(img)
        st.session_state['locations'] = None
    if 'json_file' not in st.session_state:
        #st.session_state['json_file'] = crop_editor.make_detection_canvas(st.session_state['locations'])
        st.session_state['json_file'] = None
    if 'flag_od' not in st.session_state:
        _, _, st.session_state['locations'] = OD_image(img)
        st.session_state['flag_od'] = True
    if st.session_state['locations'] is not None:#TO DO: Check OD must return locations?
        st.session_state['json_file'] = crop_editor.make_detection_canvas(st.session_state['locations'])
    #Get locations only once
    if st.session_state['json_file'] is not None:
        # Specify canvas parameters in application
        place.write("Load가 끝날 때마다 천천히 조정해야 수정이 적용됨.")
        select = place.selectbox("Tool:", ("크기 조정", "새로 그리기"))
        drawing_mode = {"크기 조정":"transform","새로 그리기":"rect"}
        bg_image = img
        shape = np.shape(bg_image)

        # Create a canvas component
        canvas = st_canvas(
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

        canvas_result = st_canvas(
            height= shape[0], # To Do :  set ratio
            width = shape[1],
            background_image= bg_image,
            initial_drawing = canvas.json_data)

        before, save, next = place.columns(3)
        images = place.empty()
        flag_b = before.button("이전")
        flag_a = next.button("다음")
        if save.button("SAVE"):
            print(st.session_state['json_file'])
            st.session_state["crop_image"] = None
            cropped_imges = []
            if canvas_result.json_data["objects"] is None:
                place.error("자를 문제가 없습니다.")
            else:
                for object in canvas_result.json_data["objects"]:
                    print(object)
                    x = object["left"]
                    y = object["top"]
                    w = object["width"]
                    h = object["height"]

                    area = (x,y,x+w,y+h)
                    cropped_img = img.crop(area)
                    cropped_img.save("test.jpg")
                    #cv2.imwrite("test.jpg",cropped_img)
                    cropped_imges.append(np.array(cropped_img))
                cv2.imwrite("test1.jpg", canvas_result.image_data)
                st.session_state["crop_image"] = cropped_imges
                page_chg('/',router)
            
                place.write("자른 문제 결과")
                if st.session_state["crop_image"] is not None:

                    images.image(cropped_imges)

        if flag_b and (st.session_state["crop_image"] is not None):
            st.session_state['sub_page'] = "first"
            page_chg('/',router)
        elif flag_a and (st.session_state["crop_image"] is not None):
            st.session_state['sub_page'] = "third"
            page_chg('/',router)
        elif flag_b or flag_a:
            place.error("자른 문제를 저장해주세요")
            page_chg('/',router)


def run_gan(place, router):
    name = st.session_state["file_name"]
    crop_images = st.session_state["crop_image"]
    gan_img = GAN_image(crop_images)

    place.subheader("손글씨 지운 사진 확인하고, 문제의 과목과 답을 입력하세요.") #Show Clear image
    place.subheader("문제의 과목과 답을 모두 입력 후에, 문제들을 저장하세요.")
    if "idx" not in st.session_state:
        st.session_state.idx = 0
        st.session_state.subject = dict()
        st.session_state.answer = dict()

    before_p, save,next_p = place.columns(3)

    if next_p.button("다음 문제"):
        if st.session_state.idx <= len(gan_img)-2:
            st.session_state.idx += 1
        else:
            st.session_state.idx = len(gan_img)-1
            st.warning("It's last problem")

    if before_p.button("이전 문제"):
        if st.session_state.idx >= 1:
            st.session_state.idx -= 1
        else:
            st.session_state.idx = 0
            st.warning("It's first problem")

    if save.button("마지막! 문제들 저장"):
        if os.path.isdir("save")==False:
            os.mkdir("save")
        
        #Check write all answer
        if (len(st.session_state['answer']) == len(gan_img)):
            for i in range(len(gan_img)):
                save_name = 'save/%s_%s_%d.jpg'%(st.session_state['user_id'], st.session_state["file_name"][:-4] , i)
                #save image to file save/
                cv2.imwrite(save_name,gan_img[i])
                # save img path in db
                #query = 'insert into problems (user_id, problem_file_name, answer) values ("%s", "%s", "%s");'%(st.session_state['user_id'], save_name, "1")
                query = 'insert into problems (user_id, problem_file_name, answer, subject) values ("%s", "%s", "%s","%s");'%(st.session_state['user_id'], save_name, st.session_state['answer'][i],st.session_state['subject'][i])
                rowcount = run_insert(query)
                if rowcount!=0:
                    st.info('문제가 저장되었습니다.')
        else:
            place.error("과목과 정답을 기재하지 않은 문제가 있습니다.")
        
    place.image(gan_img[st.session_state.idx])
    col1, col2, col3 = place.columns(3)
    sub = col1.text_input("과목은?")
    ans = col2.text_input("정답은?")

    #SAVE subject, answer
    if col3.button("과목,답 저장"):
        if (sub is not None) and (ans is not None): 
            st.session_state.subject[st.session_state.idx] = sub
            st.session_state.answer[st.session_state.idx] = ans
            print(st.session_state.answer)

    before, _, next = place.columns(3)
    if before.button("문제 자르기"):
        st.session_state['sub_page'] = "second"
        page_chg('/',router)
    elif next.button("PDF 만들기"):
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
        st.session_state["idx_p"] += 3
        page_chg('/',router)
    
    print(st.session_state["pick_problem"])

    export_as_problem_pdf = st.button("Export Problem Report")
    if export_as_problem_pdf:
        if os.path.isdir("save"):
            pdf = FPDF()
            x, y, w, h=0, 10, 120, 100
            for i in st.session_state["pick_problem"]:
                img = images[i][2]
                pdf.add_page()
                pdf.image(f"{img}", x=x, y=y, w=w, h=h)
            html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
            st.markdown(html, unsafe_allow_html = True)
    
    export_as_answer_pdf = st.button("Export Answer Report")
    if export_as_answer_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        for i in st.session_state["pick_problem"]:
            text = "Q {} : A {} \n".format(i+1,images[i][3])
            pdf.multi_cell(40,10,text)
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
        st.markdown(html, unsafe_allow_html = True)
    
    col1, col2 = place.columns(2)
    if col1.button("다시 입력하기"):
        st.session_state['sub_page'] = "third"
        page_chg('/',router)
    if col2.button("처음으로"):
        st.session_state['sub_page'] = "first"
        page_chg('/',router)

def show_images(images):
    for i in range(len(images)):
        image = Image.open(images[i][2])
        col1, col2 = st.columns(2)
        col1.write(i+1)
        col2.image(image)
        print(images[i])
    print(len(images))

@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_router():
    return stx.Router({"/":None, "/join": None})


def streamlit_run():
    router = init_router()
    router.show_route_view()
    session_init()
    
    # before login
    if st.session_state['auth_status'] != True:
        # login page
        if st.session_state['stx_router_route'] == '/' :
            if(st.session_state['after_join'] == True):
                st.info('회원가입이 완료되었습니다. 로그인을 해주세요.')
                st.session_state['after_join'] == False

            login_box = st.container()
            login_box.header('Login')
            user_id = login_box.text_input("Username")
            user_pw = login_box.text_input("Password",  type="password")

            btn1, btn2, _ = login_box.columns((1, 1.5, 5))
            if btn1.button('로그인'):
                user_id, user_name, wrong_num = login(user_id, user_pw)
                st.session_state['user_id'] = user_id
                st.session_state['user_name'] = user_name
                st.session_state['wrong_num'] = wrong_num
                print("로그인 성공")
                page_chg('/',router)

            if btn2.button('회원가입') :
                page_chg('/join', router)
        # join page
        elif st.session_state['stx_router_route'] == '/join' :
            st.title('회원가입')
            st.subheader('아이디')
            user_id = st.text_input("아이디를 입력해주세요")
            st.subheader('비밀번호')
            user_pw = st.text_input("비밀번호를 입력해주세요", type='password')
            st.subheader('이름')
            user_name = st.text_input("이름을 입력해주세요")
            
            if st.button('submit', key='join_btn'):
                result = join(user_id, user_pw, user_name)
                if result!=0 :
                    st.session_state['after_join']==True
                    page_chg('/', router)
                    
    # after login
    else:
        # sidebar content
        user_info = st.sidebar.container()
        user_info.subheader('%s님 안녕하세요!'% st.session_state['user_name'])
        if user_info.button('logout'):
            logout()
            page_chg('/',router)
        
        menu = ["실행","MakeImage","MakePDF","Answer","About", "Show All"]
        choice = st.sidebar.selectbox("Menu", menu)
        st.sidebar.text(st.session_state['prev_menu'])

        # session init
        if st.session_state['prev_menu'] != choice:
            st.session_state['prev_menu'] = choice
            #Reset cache
            for key in st.session_state.keys():
                
                if key not in ['wrong_num', 'user_name', 'user_id', 'auth_status', 'prev_menu']:
                    del st.session_state[key]
            #four session
            st.session_state['sub_page'] = 'first'

        # main content
        st.title("수학 오답 노트 생성기")

        if choice == "실행":
            st.sidebar.text("1: 파일 올리기")
            st.sidebar.text("2: 이미지 자르기")
            st.sidebar.text("3: 손글씨 지우기")
            st.sidebar.text("4: PDF 출력")

            #Use empty - Container
            place = st.empty()

            #Upload problem
            if st.session_state["sub_page"] == "first":
                first = place.container()
                upload_problem_images(first, router)
            #Crop problem
            elif st.session_state["sub_page"] == "second":
                img = st.session_state["image"]
                second = place.container()
                run_object_detection(img,second, router)
            #Erase Handwriting
            elif st.session_state['sub_page'] == "third":
                third = place.container()
                run_gan(third, router)
            #Make PDF
            elif st.session_state["sub_page"] == "fourth":
                fourth = place.container()
                fourth.subheader("4. Make Problem PDF")
                images =  run_select('SELECT * from problems where user_id="%s";' % st.session_state['user_id'])
                make_problem_pdf(fourth, router, images)

        elif choice == "Show All" :
            images =  run_select('SELECT * from problems where user_id="%s";' % st.session_state['user_id'])
            if images is not None:
                show_images(images)
        
        elif choice == "MakeImage":
            st.subheader("Upload your problem images")
            image_file = st.file_uploader("Upload Image", type=['png','jpeg','jpg'])
            if image_file is not None:
                #Get Before Image
                img = load_image(image_file)
                img = img.convert('RGB') #RGBA -> RGB
                st.subheader("Before")
                st.image(img, use_column_width = True)

                od_images,crop_images, _ = OD_image(img)
                seg_img = seg_image(crop_images)
                gan_img = GAN_image(seg_img)

                flag_od = st.checkbox("Object Detection")
                flag_seg = st.checkbox("segmentation")
                flag_gan = st.checkbox("GAN")

                if flag_od:
                    st.subheader("Object Detection")
                    st.image(od_images, use_column_width = True)
                if flag_seg:
                    st.subheader('Segmentation')
                    st.image(seg_img[0],use_column_width = True)
                if flag_gan:
                    st.subheader("GAN")
                    st.image(gan_img[0], use_column_width = True)

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