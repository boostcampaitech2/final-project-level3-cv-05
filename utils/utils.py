import streamlit as st
import mysql.connector
import extra_streamlit_components as stx
import os

conn = mysql.connector.connect(**st.secrets["mysql"])
print("connected" ,conn.is_connected())

@st.cache(ttl=600)
def run_select(query):
    cur = conn.cursor()
    cur.execute(query)
    return cur.fetchall()

@st.cache(ttl=600)
def run_insert(query):
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    return cur.rowcount


def page_chg(key, router):
    st.session_state['page'] = key
    router.route(key)


def login(user_id, user_pw):
    rows = run_select('SELECT * from users where user_id="%s" and user_pw="%s";' % (user_id, user_pw))
    print(rows)
    if(rows):
        
        user_id, _, user_name, wrong_num = rows[0]
        st.session_state['auth_status'] = True
        return user_id, user_name, wrong_num
        
    else:
        st.error('Username/password is incorrect')
        st.session_state['auth_status'] = False
        return False, False, False


def logout():
    st.session_state['auth_status'] = False
    

def join(user_id, user_pw, user_name):
    rows = run_select('SELECT * from users where user_id="%s";' % user_id)
    if (rows):
        st.error('같은 아이디가 존재합니다.')
        return 0
    else:
        query = 'insert into users values ("%s", "%s", "%s", %d);'%(user_id, user_pw, user_name, 0)

        rowcount = run_insert(query)
        return rowcount
        

def mkdir(dir_):
    if os.path.isdir(dir_) == False: #Change path
        os.mkdir(dir)