from PIL import Image
from fpdf import FPDF

import streamlit as st
import extra_streamlit_components as stx
import os
import base64
import cv2

#streamlit run server.py --server.address=127.0.0.1
#pip install extra-streamlit-components
#pip install mysql-connector

import cropper
from utils.utils import *
from utils.model import *

#wide
st.set_page_config(layout="wide")

# Fxn
@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img 

#(Delete) Fxn to Save Uploaded File to Directory
def save_uploaded_file(uploadfile):
    if(os.path.isdir("math_data") == False): #Change path
        os.mkdir("math_data")

    with open(os.path.join("math_data",uploadfile.name),'wb') as f:
        f.write(uploadfile.getbuffer())
    return st.success("Upload file :{} in Server".format(uploadfile.name))


#(Delete) Fxn to Save After File
def save_after_file(file,name):
    if(os.path.isdir("new_data")==False):
        os.mkdir("new_data")

    file.save('./new_data/{}'.format(name),'png')
    return st.success("Upload file :{} in Server".format(name))

#Download PDF
def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_router():
    return stx.Router({"/":None, "/join": None})

def streamlit_run():
    router = init_router()
    router.show_route_view()
    #session initialize
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    if 'auth_status' not in st.session_state:
        st.session_state['auth_status'] = None
    if 'after_join' not in st.session_state:
        st.session_state['after_join'] = None
    if 'prev_menu' not in st.session_state:
        st.session_state['prev_menu'] = ''
    
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
        
        menu = ["All","MakeImage","Answer","About", "Show All"]
        choice = st.sidebar.selectbox("Menu",menu)
        st.sidebar.text(st.session_state['prev_menu'])

        # session init
        if st.session_state['prev_menu'] != choice:
            st.session_state['prev_menu'] = choice
            #Reset cache
            for key in st.session_state.keys():
                if key not in ['wrong_num', 'user_name', 'user_id', 'auth_status', 'prev_menu']:
                    del st.session_state[key]

        # main content
        st.title("Math wrong answer editor")

        if choice == "All":
            st.header("All part")
            st.subheader("1. Upload your problem images")


            with st.form("Upload"):
                image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
                submit = st.form_submit_button("Upload")

            if image_file and submit is not None:
                img = load_image(image_file) #Get Image
                img = img.convert('RGB') #RGBA -> RGB
                st.image(img)

            if submit:
                st.session_state["OD_button"] = False
                st.session_state["OD_show_button"] = False
                st.session_state["GAN_button"] = False
                st.session_state["idx"] = 0

            
            st.subheader("2. Check wrong image, and you can edit")
            
            if 'OD_button' not in st.session_state:
                st.session_state["OD_button"] = False
            if st.button("Crop"):
                st.session_state["OD_button"] = True
            if st.session_state["OD_button"]:
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
                    cropper.crop(image_file)

                if "OD_show_button" not in st.session_state:
                    st.session_state["OD_show_button"] = False
                
                if st.button("Show"):
                    st.session_state["OD_show_button"] = True
                
                if st.session_state["OD_show_button"]:
                    st.image(crop_images) 

            st.subheader("3. Clear handwriting")

            if 'GAN_button' not in st.session_state:
                st.session_state["GAN_button"] = False
            if st.button("Clear"):
                st.session_state["GAN_button"] = True
            if st.session_state["GAN_button"]:
                gan_img = GAN_image(crop_images)

                st.subheader("Check Final Image & Save") #Show Clear image
                if "idx" not in st.session_state:
                    st.session_state["idx"] = 0

                before, next, save = st.columns(3)

                if next.button("Next"):
                    if st.session_state["idx"] <= len(gan_img)-2:
                        st.session_state["idx"] += 1
                    else:
                        st.session_state["idx"] = len(gan_img)-1
                        st.warning("It's last problem")

                if before.button("Before"):
                    if st.session_state["idx"] >= 1:
                        st.session_state["idx"] -= 1
                    else:
                        st.session_state["idx"] = 0
                        st.warning("It's first problem")

                if save.button("Save"):
                    if os.path.isdir("save")==False:
                        os.mkdir("save")
                    

                    for i in range(len(gan_img)):
                        save_name = 'save/%s_%s_%d.jpg'%(st.session_state['user_id'], image_file.name[:-4] , i)
                        cv2.imwrite(save_name,gan_img[i])
                        # save img path in db
                        query = """insert into problems (user_id, problem_file_name, answer) values ('%s', '%s', '%s');"""%(st.session_state['user_id'], save_name, '1')
                        rowcount = run_insert(query)
                        if rowcount!=0:
                            st.info('문제가 저장되었습니다.')
                #Show image
                st.image(gan_img[st.session_state["idx"]])

            
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

        elif choice == "Show All" :
            images =  run_select('SELECT * from problems where user_id="%s";' % st.session_state['user_id'])
            for image_f in images:
                image = Image.open(image_f[2])
                st.image(image)
        #Will Delete After Modeling
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