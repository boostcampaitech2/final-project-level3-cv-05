import streamlit as st
import os

import pandas as pd
import csv
from PIL import Image
# from fpdf import FPDF
import base64

#streamlit run app.py --server.address=127.0.0.1
#이렇게 하면 브라우저가 Local로 띄어짐.

# Fxn
@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 

#Fxn to Save answer
def save_results(results_df,button_press,image_file,p_name,answer):
	results_df.at[button_press,'File name'] = image_file.name
	results_df.at[button_press,'Nick name'] = p_name
	results_df.at[button_press,'Answer'] = answer
	results_df.to_csv('answer.csv',index=None)
	return None

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

def main():
	result_df = load_data()
	button_press = 0
	#button_press = len(result_df)

	st.title("Math wrong answer editor")

	menu = ["MakeImage","MakePDF","Answer","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "MakeImage":
		st.subheader("Upload your problem images")
		image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
		if image_file is not None:
			#upload Image
			img = load_image(image_file)
			result_df = load_data()

			col1, col2 = st.columns(2)

			#original = Image.open(img).convert("RGB")
			col1.header("Before")
			col1.image(img, use_column_width = True)

			new = img.convert('LA') #model된 걸로 바꾸기

			col2.header("After")
			col2.image(new,use_column_width=True)

			#button_press = len(result_df)
			st.write(image_file.name)

			#write name
			p_name = st.text_input('Write nickname')

			#write anwer
			answer = st.text_input('Write answer')

			#saving file
			if st.button("Save"):
				button_press += 1
				st.write(button_press)
				save_results(result_df, button_press, image_file, p_name, answer)
				save_uploaded_file(image_file)
				save_after_file(new,p_name)
			#Done?
			if st.button("Are you done?"):
				save_uploaded_csv(result_df)
	elif choice == "MakePDF":
		#PDF 구현
		st.text("Show")
		
		#Save pdf
	elif choice == "Answer":
		st.subheader("Input name to find answer")

		df = pd.read_csv("answer.csv")
		st.write(df)
	else:
		st.subheader("About")
		st.text("수학 오답 노트 편집기")
		st.text("친구구했조")
		st.text("Freinds")



if __name__ == '__main__':
	main()
