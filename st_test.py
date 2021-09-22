# import libraries
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import pandas as pd
import keras
import os
import time
from stqdm import stqdm

#defined setting
WIDTH = 128
model_path = r'model'
label = ['AdvancedGlaucoma','EarlyGlaucoma','Normal']

def load_model(model_name):
    model = keras.models.load_model(os.path.join(model_path , model_name))
    return model

# set function
def img2data(path,label):
    rawImgs = []
    labels = []
    name = []
    c = 0
    for imagePath in (path):
        for item in stqdm(os.listdir(imagePath)):
            file = os.path.join(imagePath, item)
            name.append(item)
            c+=1
            l = item.split('_')[0]
            if l == label[0]: labels.append([1,0,0])
            elif l == label[1]: labels.append([0,1,0])
            elif l == label[2]: labels.append([0,0,1])
            img = cv2.imread(file , cv2.COLOR_BGR2RGB)
            img = cv2.resize(img ,(WIDTH,WIDTH))
            rawImgs.append(img)
            time.sleep(0.03)
    return rawImgs, labels, name

def get_results(model,x,label):
    results = model.predict(x)
    predict = []
    for i in range(len(results)) :
        predict.append(label[np.argmax(results[i])])
    return predict

def show_data(x_test, pics_name, prediction):
    name_pic = st.selectbox('select pics_name',pics_name)
    name_pic = pics_name[0]
    position_pic = pics_name.index(name_pic)
    st.image(cv2.cvtColor(x_test[position_pic], cv2.COLOR_BGR2RGB),width=400)
    st.write('Prediction of selected picture is : ',prediction[position_pic])


def check(x_test, name_test, predict_aug):
    # select picture to check
    check = st.button('Please click this massage if you want to see specific picture and result')
    if check:
        show_data(x_test, name_test, predict_aug)

def get_dirname():
    # Set up tkinter
    root = tk.Tk()
    root.withdraw()
    
    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)
    dirname = st.sidebar.text_input('Selected folder:', filedialog.askdirectory(master=root))
    return dirname


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
    
    # Folder picker button
    st.sidebar.title('Folder Picker')
    st.sidebar.write('Please select a folder of image:')
    clicked = st.sidebar.button('Folder Picker')
    return clicked

#@st.cache(suppress_st_warning=True)
def main_computational(dirname, model):
    start_time = time.time()
        
    x_test, y_test, name_test = img2data([dirname],label)
    x_test = np.array(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255
    
    predict_aug = get_results(model,x_test,label)
    dict = {'pics_name': name_test, 'prediction': predict_aug} 
    df = pd.DataFrame(dict)
    st.title('Results')
    st.dataframe(df)
    st.write("Total run time : --- %s seconds ---" % (time.time() - start_time))
    return x_test, name_test, predict_aug

def main():
    # Load model
    model = load_model('MbN_AEN_N_aug')
    show_strict = strict_part()
    
    if show_strict:
        dirname = get_dirname() #
        x_test, name_test, predict_aug = main_computational(dirname, model)
        #check(name_test, predict_aug)
        show_data(x_test, name_test, predict_aug)

if __name__ == '__main__':
	main()