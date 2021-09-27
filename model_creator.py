#import requirement
import streamlit as st
import numpy as np
import cv2
import time
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from zipfile import ZipFile
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

# -------------------------------- Setup --------------------------------

# Set up hyperparameters.
LEARNING_RATES = 0.0001
BATCH_SIZE = 32
EPOCHS = 100
WIDTH = 128
N_THAMSAM = 3
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATES)
LOSS = 'categorical_crossentropy'


# -------------------------------- Load image from zip file --------------------------------
def img2data_zip(zip_file):
    rawImgs = []
    labels = []
    class_list = []
    archive = ZipFile(zip_file, 'r')
    files = archive.infolist()
    name_list = archive.namelist()
    for i in range(len(name_list)):
        name = name_list[i]
        img_class = name.split('/')[0]
        if name[-4:]!='.png' and name[-4:]!='.jpeg':
            class_list.append(img_class)
            
    for i in range(len(name_list)):
        name = name_list[i]
        img_class = name.split('/')[0]
        if name[-4:]=='.png' or name[-4:]=='.jpeg':
            img_data = archive.open(files[i])
            PIL_img = Image.open(img_data)
            img = np.array(PIL_img) # change PIL to numpy
            img = img[:, :, ::-1].copy() 
            img = cv2.resize(img ,(WIDTH,WIDTH))
            rawImgs.append(img)
            for i in range(len(class_list)):
                if img_class == class_list[i]: labels.append(i)
    labels = to_categorical(labels, dtype=int)
            
    return rawImgs, labels, class_list

# -------------------------------- Defined Model --------------------------------
# model architecture
def build_model(WIDTH,NUM_CLASS):
    model = keras.Sequential([
        keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(WIDTH, WIDTH, 3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128,(3,3) , activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2 )),
        keras.layers.Dense(16),
        keras.layers.Flatten(),

        keras.layers.Dense(NUM_CLASS, activation='softmax') #softmax for one hot . . # sigmoid for 0/1
    ])
    return model

# -------------------------------- Defined function --------------------------------
def get_results(x, y, label):
    results = model.predict(x)
    predict = []
    behavior = []
    i=0
    for result in results :
        predict.append(label[np.argmax(result)])
        behavior.append(label[np.argmax(y[i])])
        i+=1
    return predict,behavior

def scoring(y_true,y_predict):
    st.text('ACC: ' +str(accuracy_score(y_true,y_predict)))
    st.text('MCC: ' +str(matthews_corrcoef(y_true,y_predict)))
    st.text('F1 SCORE: ' +str(f1_score(y_true,y_predict,average='weighted')))
    st.text('PRECISION: ' +str(precision_score(y_true,y_predict,average='weighted')))
    st.text('RECALL: ' +str(recall_score(y_true,y_predict,average='weighted')))
    
def main():
    # read zip file
    train_zip = st.sidebar.file_uploader('Pls upload train zip file',type='zip')
    test_zip = st.sidebar.file_uploader('Pls upload test zip file',type='zip')
    x_train, y_train, class_list = img2data_zip(train_zip)
    x_test, y_test, class_list = img2data_zip(train_zip)
    
    # manage file
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    # show file size
    st.text(x_train.shape,y_train.shape,x_test.shape, y_test.shape)
    
    acc = []
    acc_val = []
    loss_val = []
    for i in range(N_THAMSAM):
        model = build_model(WIDTH,len(class_list))
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics= ['accuracy'])

        history = model.fit(x_train, y_train ,batch_size=BATCH_SIZE, epochs=EPOCHS,validation_data=(x_test, y_test))

        acc_list = history.history['accuracy']
        acc_val_list = history.history['val_accuracy']
        loss_val_list = history.history['val_loss']
        acc.append(acc_list[-1])
        acc_val.append(acc_val_list[-1])
        loss_val.append(loss_val_list[-1])
        
    st.text('avg_acc : ',np.mean(np.array(acc)))
    st.text('std_acc : ',np.std(np.array(acc)))
    st.text('avg_acc_val : ',np.mean(np.array(acc_val)))
    st.text('std_acc_val : ',np.std(np.array(acc_val)))
    st.text('avg_loss_val : ',np.mean(np.array(loss_val)))
    st.text('std_loss_val : ',np.std(np.array(loss_val)))
    
    predict,behavior = get_results(x_test, y_test, class_list)
    st.text(confusion_matrix(behavior, predict))
    scoring(behavior, predict)