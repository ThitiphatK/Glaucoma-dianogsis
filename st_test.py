# import libraries
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import pandas as pd
import keras
import os
from stqdm import stqdm

#defined setting
WIDTH = 128
path = r'model'
label = ['AdvancedGlaucoma','EarlyGlaucoma','Normal']
model_aug = keras.models.load_model(os.path.join(path , 'MbN_AEN_N_aug'))

# Set up tkinter
root = tk.Tk()
root.withdraw()


# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

# Folder picker button
st.title('Folder Picker')
st.write('Please select a folder of image:')
clicked = st.button('Folder Picker')
st.sidebar.write('Example image')
st.sidebar.image(
    'https://www.reviewofoptometry.com/CMSImagesContent/2011/11/030_RO1111_F1.gif',
    width=200,
)

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
            l = imagePath.split('/')[1]
            if l == label[0]: labels.append([1,0,0])
            elif l == label[1]: labels.append([0,1,0])
            elif l == label[2]: labels.append([0,0,1])
            img = cv2.imread(file , cv2.COLOR_BGR2RGB)
            img = cv2.resize(img ,(WIDTH,WIDTH))
            rawImgs.append(img)
    return rawImgs, labels, name

def get_results(model,x,label):
    results = model.predict(x)
    predict = []
    for i in range(len(results)) :
        predict.append(label[np.argmax(results[i])])
    return predict

if clicked:
    dirname = st.text_input('Selected folder:', filedialog.askdirectory(master=root))
    
    x_test, y_test, name_test = img2data([dirname],label)
    x_test = np.array(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255
    
    predict_aug = get_results(model_aug,x_test,label)
    dict = {'pics_name': name_test, 'prediction': predict_aug} 
    df = pd.DataFrame(dict)
    st.dataframe(df)