from PIL import Image

import streamlit as st
import extra_streamlit_components as stx
from utils.utils import *
from step import upload_problem_images,run_object_detection,run_gan,make_problem_pdf

#streamlit run server.py --server.address=127.0.0.1
#이렇게 하면 브라우저가 Local로 띄워짐.

#wide
st.set_page_config(layout="wide")

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

def show_images(images):
    for i in range(len(images)):
        image = Image.open(images[i][2])
        col1, col2 = st.columns(2)
        col1.write(i+1)
        col2.image(image)

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
        
        menu = ["실행","MakePDF","Answer","About", "Show All"]
        choice = st.sidebar.selectbox("Menu", menu)
        st.sidebar.text(st.session_state['prev_menu'])

        # session reset
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
            else:
                st.warning("저장되어있는 문제가 없습니다.")
        elif choice == "Answer":
            st.text("Show")
        else:
            st.subheader("About")
            st.text("수학 오답 노트 편집기")
            st.text("친구구했조")
            st.text("Freinds")


if __name__ == '__main__':
    streamlit_run()