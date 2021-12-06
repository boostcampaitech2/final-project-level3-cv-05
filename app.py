import streamlit as st
import os

import pandas as pd
import csv
from PIL import Image
import glob 

ip = '101.101.219.102:' #change server ip
upload_dir = '/opt/ml/math_data' #change file
download_dir = '/opt/ml/math_data' #model 돌린 후, 저장될 폴더
local_dir = 'C:\\Users\\cac73' #local dir
dir_name = "math_data" # download_dir 랑 같은 이름으로 하기.

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
	command = "scp " + "answer.csv" + " root@" + ip + upload_dir
	os.system(command)
	return st.success("Save Answer : To Show Click Answer on Menu")

#Fxn to Save Uploaded File to Directory
def save_uploaded_file(uploadedfile):
	command = "scp " + uploadedfile.name + " root@" + ip + upload_dir
	os.system(command)
	return st.success("Upload file :{} in Server".format(uploadedfile.name))

#Fxn to Save Downloaded File 
def save_download_file():
	command = "scp -P 2229 -i ~/.ssh/key -r root@" + ip + download_dir + " " + local_dir 
	os.system(command)
	return st.success("Download file : Data in your local {}".format(local_dir))


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
			
			#Done?
			if st.button("Are you done?"):
				save_uploaded_csv(result_df)


	elif choice == "MakePDF":
		if st.button("Download make image"):
			save_download_file()
		
		st.subheader("Print PDF")

		#dir 에서 사진 파일만 가져오기.
		file_dir = local_dir + "\\" + dir_name
		file_list = os.listdir(file_dir)
		img_files = [file for file in file_list if file.endswith(('png','jpeg','jpg'))]
		
		st.write(img_files)

		image = load_image(file_dir + "\\" + img_files[0])

		images = []
		for idx,img in enumerate(img_files):
			images.append(load_image(file_dir + "\\" + img_files[idx]))

		#To do Change 
		cols = st.columns(4)
		for idx, img in enumerate(images):
			cols[idx].image(images[idx],use_column_width=True)
		
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
