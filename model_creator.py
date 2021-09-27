#import requirement
import streamlit as st
import numpy as np
import cv2
import time
import keras
import tensorflow as tf
import collections
from tensorflow.keras.utils import to_categorical
from zipfile import ZipFile
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

# -------------------------------- Setup --------------------------------

# Set up hyperparameters.
st.sidebar.title('Setting Program')
hyper_set = st.sidebar.selectbox('Setting hyper parameter',['Adjust','Default'])
if hyper_set == 'Default':
    LEARNING_RATES = 0.0001
    BATCH_SIZE = 32
    EPOCHS = 100
    WIDTH = 128
    N_THAMSAM = 1
    
else :
    LEARNING_RATES = st.sidebar.slider('learning rates', min_value=0.0001, max_value=0.001, step=0.0001)
    BATCH_SIZE = st.sidebar.selectbox('batch size', [8,16,32])
    EPOCHS = st.sidebar.slider('epochs', min_value=10, max_value=100, step=10)
    WIDTH = st.sidebar.selectbox('width', [64,128,256])
    N_THAMSAM = st.sidebar.slider('amount of re-training', min_value=1, max_value=5, step=1)

st.title('------- Hyper parameter setting -------')
st.markdown('Learning rates : '+str(LEARNING_RATES))
st.markdown('Batch size : '+str(BATCH_SIZE))
st.markdown('Epochs : '+str(EPOCHS))
st.markdown('Width : '+str(WIDTH))
st.markdown('Amount of re-training : '+str(N_THAMSAM))

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATES)
LOSS = 'categorical_crossentropy'


# -------------------------------- Load image from zip file --------------------------------
def img2data_zip(zip_file):
    rawImgs = []
    labels = []
    class_list = []
    each_class_appearance = []
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
            each_class_appearance.append(img_class)
    labels = to_categorical(labels, dtype=int)
            
    return rawImgs, labels, class_list, each_class_appearance

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
def get_results(y_pred, y, label):
    results = y_pred
    predict = []
    behavior = []
    i=0
    for result in results :
        predict.append(label[np.argmax(result)])
        behavior.append(label[np.argmax(y[i])])
        i+=1
    return predict,behavior

def scoring(y_true,y_predict):
    st.markdown('ACC: ' +str(accuracy_score(y_true,y_predict)))
    st.markdown('MCC: ' +str(matthews_corrcoef(y_true,y_predict)))
    st.markdown('F1 SCORE: ' +str(f1_score(y_true,y_predict,average='weighted')))
    st.markdown('PRECISION: ' +str(precision_score(y_true,y_predict,average='weighted')))
    st.markdown('RECALL: ' +str(recall_score(y_true,y_predict,average='weighted')))
    
def main():
    # read zip file
    st.sidebar.title('Please upload zip file of train image: ')
    train_zip = st.sidebar.file_uploader('Choose train zip file',type='zip')
    st.sidebar.title('Please upload zip file of test image: ')
    test_zip = st.sidebar.file_uploader('Choose test zip file',type='zip')
    click = st.sidebar.button('Click this button to create model')
    if click:
        x_train, y_train, class_list, train_appearance = img2data_zip(train_zip)
        x_test, y_test, class_list, test_appearance = img2data_zip(test_zip)

        counter_train=collections.Counter(train_appearance)
        counter_test=collections.Counter(test_appearance)
        
        # manage file
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # show file details
        st.title('------------- Info of zip file -------------')
        st.markdown('Class exist in file : '+ str(class_list))
        st.markdown('Each class appearance in train file :'+str(dict(counter_train)))
        st.markdown('Each class appearance in test file :'+str(dict(counter_test)))
        st.markdown('Train size : '+ str(x_train.shape))
        st.markdown('Test size : '+ str(x_test.shape))

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
        
        st.title('------------ Training results ------------')
        st.markdown('avg_acc : '+str(np.mean(np.array(acc))))
        st.markdown('std_acc : '+str(np.std(np.array(acc))))
        st.markdown('avg_acc_val : '+str(np.mean(np.array(acc_val))))
        st.markdown('std_acc_val : '+str(np.std(np.array(acc_val))))
        st.markdown('avg_loss_val : '+str(np.mean(np.array(loss_val))))
        st.markdown('std_loss_val : '+str(np.std(np.array(loss_val))))

        st.title('--------- Lastest model results ---------')
        y_pred = model.predict(x_test)
        predict,behavior = get_results(y_pred, y_test, class_list)
        st.text(confusion_matrix(behavior, predict))
        scoring(behavior, predict)
    
if __name__ == '__main__':
	main()