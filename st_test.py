# import libraries
import streamlit as st
#import tkinter as tk
#from tkinter import filedialog
import cv2
import numpy as np
import pandas as pd
import keras
import os
import time
from stqdm import stqdm
from zipfile import ZipFile
from PIL import Image

# ----------------- Streamlit page setting -----------------
st.set_page_config(page_title="GLAUCOMA DIAGNOSSIS", 
                page_icon=":eye:",
                layout="wide",
                initial_sidebar_state="expanded"
                )

#defined setting
WIDTH = 128
model_path = r'model'
label = ['AdvancedGlaucoma','EarlyGlaucoma','Normal']

#@st.cache(suppress_st_warning=True)
def strict_part():
    # Details of Application
    st.title('Glaucoma diagnosis Application')
    
    st.markdown("""
    This app provides an image classification model for glaucoma diagnosis that was trained using **MobileNetV2** 
    and a dataset of fundus images from **Harvard Dataverse**. [ Note that! the dataset name is Machine learn for glaucoma, 
    and it was provided by Kim, Ungsoo ].
    * **Python libraries:** streamlit, tkinter, cv2, numpy, pandas, keras, os, stqdm
    * **Data source:** [Harvard Dataverse](https://doi.org/10.7910/DVN/1YRRAC).
    """)
    
    #example image
    st.sidebar.title('Example image')
    st.sidebar.image(
        'https://www.reviewofoptometry.com/CMSImagesContent/2011/11/030_RO1111_F1.gif',
        width=200,
    )
    st.sidebar.markdown('[Image credit](https://www.reviewofoptometry.com/CMSImagesContent/2011/11/030_RO1111_F1.gif)')
    
    with st.sidebar.beta_expander('Contact'):
        st.write("I'd love your feedback :smiley:")
    
# ----------------- Defined function -----------------

def load_model(model_name):
    model = keras.models.load_model(os.path.join(model_path , model_name))
    return model

#def get_dirname():
#    # Set up tkinter
#    root = tk.Tk()
#    root.withdraw()
#    
#    # Make folder picker dialog appear on top of other windows
#    root.wm_attributes('-topmost', 1)
#    dirname = st.text_input('Selected folder:', filedialog.askdirectory(master=root))
#    return dirname

# set function
#def img2data_directory(path):
#    rawImgs = []
#    name = []
#    for imagePath in (path):
#        for item in stqdm(os.listdir(imagePath)):
#            file = os.path.join(imagePath, item)
#            name.append(item)
#            img = cv2.imread(file , cv2.COLOR_BGR2RGB)
#            img = cv2.resize(img ,(WIDTH,WIDTH))
#            rawImgs.append(img)
#            time.sleep(0.03)
#    return rawImgs, name

def img2data_zip(zip_file):
    rawImgs = []
    name = []
    archive = ZipFile(zip_file, 'r')
    files = archive.infolist()
    name_list = archive.namelist()
    for i in range(len(name_list)):
        img_data = archive.open(files[i])
        #img = cv2.imread(img_data , cv2.COLOR_BGR2RGB)
        PIL_img = Image.open(img_data)
        img = np.array(PIL_img) # change PIL to numpy
        img = img[:, :, ::-1].copy() 
        img = cv2.resize(img ,(WIDTH,WIDTH))
        rawImgs.append(img)
        name.append(name_list[i])
        time.sleep(0.03)
    return rawImgs, name

def img2data_image(file):
    rawImgs = []
    name = []
    PIL_img = Image.open(file)
    img = np.array(PIL_img) # change PIL to numpy
    img = img[:, :, ::-1].copy() 
    img = cv2.resize(img ,(WIDTH,WIDTH))
    rawImgs.append(img)
    name = str(file)
    return rawImgs, name
    
def choose_type_input():
    st.title('Select input type')
    input_type = st.selectbox('Select',['image','zip'])
    if input_type:
        if input_type=='image':
            input = st.file_uploader('Pls upload image file',type=['png','jpeg'])
        elif input_type=='zip':
            input = st.file_uploader('Pls upload zip file',type='zip')
        #else : 
        #    input = get_dirname()
    return input_type, input

def get_results(model,x,label):
    results = model.predict(x)
    predict = []
    for i in range(len(results)):
        predict.append(label[np.argmax(results[i])])
    return predict

def show_data(x_test, pics_name, prediction):
    name_pic = st.selectbox('select pics_name', pics_name)
    #name_pic = pics_name[0]
    if name_pic:
        position_pic = pics_name.index(name_pic)
        st.image(cv2.cvtColor(x_test[position_pic], cv2.COLOR_BGR2RGB),width=400)
        st.write('Prediction of selected picture is : ',prediction[position_pic])


def check(x_test, pics_name, prediction):
    # select picture to check
    check = st.button('Please click this massage if you want to see specific picture and result')
    if check:
        show_data(x_test, pics_name, prediction)

#@st.cache(suppress_st_warning=True)
def main_computational(input_type, input, model):
    start_time = time.time()
    
    if input_type=='image':
        x_test, name_test = img2data_image(input)
    elif input_type=='zip':
        x_test, name_test = img2data_zip(input)
    #else:
    #    x_test, name_test = img2data_directory([input])
    
    x_test = np.array(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255
    
    predicted = get_results(model,x_test,label)
    dict = {'pics_name': name_test, 'prediction': predicted} 
    df = pd.DataFrame(dict)
    st.title('Results')
    st.dataframe(df)
    st.write("Total run time : --- %s seconds ---" % (time.time() - start_time))
    return x_test, name_test, predicted, True

# ----------------- Program part -----------------

def main():
    strict_part()
    
    # Load model
    model = load_model('MbN_AEN_N_aug')
    # select input type
    input_type, input = choose_type_input()
    done_select = st.button('RUN APPLICATION!!')
    if done_select :
        x_test, name_test, predict_aug, done_process = main_computational(input_type, input, model)
        if done_process:
            #check(x_test, name_test, predict_aug)
            show_data(x_test, name_test, predict_aug)

if __name__ == '__main__':
	main()
