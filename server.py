from PIL import Image

import streamlit as st
import extra_streamlit_components as stx
from utils.utils import *
from step import (load_image, upload_problem_images, make_problem_pdf,
                  run_object_detection, run_gan, run_seg)

#streamlit run server.py --server.address=127.0.0.1
#이렇게 하면 브라우저가 Local로 띄워짐.

#wide
st.set_page_config(layout="wide")

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

def show_images(images):
    for i in range(0,len(images),2):
        col1, col2 = st.columns(2)
        try:
            img1 = load_image(images[i][2])
        except IndexError:
            pass
        else:
            col1.image(img1)

        try:
            img2 = load_image(images[i+1][2])
        except IndexError:
            pass
        else:
            col2.image(img2)

@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_router():
    return stx.Router({"/":None, "/join": None})

def streamlit_run():
    router = init_router()
    router.show_route_view()
    session_init()
    
    # before login
    sess_state = st.session_state
    if sess_state['auth_status'] != True:
        # login page
        if sess_state['stx_router_route'] == '/' :
            if(sess_state['after_join'] == True):
                st.info('회원가입이 완료되었습니다. 로그인을 해주세요.')
                sess_state['after_join'] == False

            login_box = st.container()
            login_box.header('Login')
            user_id = login_box.text_input("Username")
            user_pw = login_box.text_input("Password",  type="password")

            btn1, btn2, _ = login_box.columns((1, 1.5, 5))
            if btn1.button('로그인'):
                user_id, user_name, wrong_num = login(user_id, user_pw)
                sess_state['user_id'] = user_id
                sess_state['user_name'] = user_name
                sess_state['wrong_num'] = wrong_num
                print("로그인 성공")
                page_chg('/',router)

            if btn2.button('회원가입') :
                page_chg('/join', router)
        # join page
        elif stx_router_route == '/join' :
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
                    sess_state['after_join']==True
                    page_chg('/', router)
                    
    # after login
    else:
        # sidebar content
        user_info = st.sidebar.container()
        user_info.subheader('%s님 안녕하세요!'% sess_state['user_name'])
        if user_info.button('logout'):
            logout()
            page_chg('/',router)
        
        menu = ["실행","문제 PDF 만들기","전체 문제 보기", "About"]

        choice = st.sidebar.selectbox("Menu", menu)
        st.sidebar.text(sess_state['prev_menu'])

        # session reset
        if sess_state['prev_menu'] != choice:
            sess_state['prev_menu'] = choice
            #Reset cache
            for key in sess_state.keys():
                if key not in ['wrong_num', 'user_name', 'user_id', 'auth_status', 'prev_menu']:
                    del sess_state[key]
            #four session
            sess_state['sub_page'] = 'first'

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
            sub_page = sess_state["sub_page"]
            if sub_page == "first":
                first = place.container()
                upload_problem_images(first, router)
            #Crop problem
            elif sub_page == "second":
                img = sess_state["image"]
                second = place.container()
                run_object_detection(img,second, router)
            #Erase Handwriting
            elif sub_page == "third":
                third = place.container()
                #Segmentation & gan
                run_gan(third, router)
            #Make PDF
            elif sub_page == "fourth":
                fourth = place.container()
                fourth.subheader("4. 문제를 선택하고, 문제지와 답지를 PDF로 받으세요!")

                images =  run_select('SELECT * from problems where user_id="%s";' % sess_state['user_id'])
                make_problem_pdf(fourth, router, images, True)
        elif choice == "문제 PDF 만들기":
            images =  run_select('SELECT * from problems where user_id="%s";' % sess_state['user_id'])
            make_problem_pdf(st, router, images,False)
        elif choice == "전체 문제 보기" :
            images =  run_select('SELECT * from problems where user_id="%s";' % sess_state['user_id'])
            if images is not None:
                show_images(images)
            else:
                st.warning("저장되어있는 문제가 없습니다.")
        else:
            st.subheader("About")
            st.text("수학 오답 노트 편집기")
            st.text("친구구했조")
            st.text("Freinds")


if __name__ == '__main__':
    streamlit_run()