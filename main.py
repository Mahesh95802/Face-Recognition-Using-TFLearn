import warnings

from numpy.core.fromnumeric import argmax
warnings.filterwarnings('ignore')

import tensorflow as tf
import tflearn
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
import os
import pickle
from random import shuffle
from tqdm import tqdm

print(tf.__version__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db = os.path.join(BASE_DIR, "DataBase")

def CreateDataWithLabels():
    print("Starting to Create and Label the Data....")
    label_ids = {}
    i_id = 0
    data = []
    for root, dirs, files in os.walk(db):
        label_array = [ 0 for i in range(len(os.listdir(db)))]
        #print(label_array)
        label_array[i_id] = 1
        #print(label_array)
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if not label in label_ids:
                    img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img_data = cv2.resize(img_data, (200,200))
                    label_ids[label] = i_id
                    i_id += 1
                data.append([np.array(img_data), np.array(label_array)])
                #print(np.array(img_data)," - ",label_ids[label]," - ",label," - ",path)
                print(label_ids[label]," - ",label)
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)
    shuffle(data)  
    print("Creation and Labeling of Data Completed.")
    return data

def train(ret):
    data = CreateDataWithLabels()
    n = len(os.listdir(db))
    print("Seperating the Data into Training and Testing DataSets.....")
    #Number of Images used for Training (NIUT)
    NIUT = int(len(data) * 0.8) 
    #Training DataSet
    train = data[:NIUT]  
    #Testing Dataset
    test = data[NIUT:]
    X_train = np.array([i[0] for i in train]).reshape(-1,200,200,1)
    print(X_train.shape)
    y_train = [i[1] for i in train]
    #print(y_train)
    X_test = np.array([i[0] for i in test]).reshape(-1,200,200,1)
    print(X_test.shape)
    y_test = [i[1] for i in test]
    #print(y_test)
    print("Data Seperated into Training and Testing DataSets.")
    tf.reset_default_graph()
    convnet = input_data(shape=[200,200,1])
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    # 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, n, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
    model = tflearn.DNN(convnet, tensorboard_verbose=1)
    print("Starting The Training of the Model")
    model.fit(X_train, y_train, n_epoch=12, validation_set=(X_test, y_test), show_metric = True, run_id="FRS" )
    print("Model Trained Successfully.")
    if (ret == "stay"):
        pass
    elif (ret == "send"):
        return model
    #model.save('facerecognition.h5')
    #print("Model Saved Successfully.")

def RegisterNewFace():
    print("Starting the Registration Process...")
    name = input("Name of the new Face: ")
    face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_alt2.xml")
    captureDevice = cv2.VideoCapture(0) 
    path = db+"\\"+name
    os.mkdir(path)
    img_id = 0
    while(True):
        ret, frame = captureDevice.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
        for (x, y, w, h) in faces:
            print(x, y, w, h)
            #Draw the Box
            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
            cropped_face = gray[y:y+h,x:x+w]
            if cropped_face is not None:
                img_id+=1
                face = cv2.resize(cropped_face, (200,200))
                file_name_path = path+"\\"+str(img_id)+'.jpg'
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (200,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
                cv2.imshow("Cropped_Face", face)
        cv2.imshow('frame', frame)
        if (cv2.waitKey(20) & 0xFF == ord('q')) or int(img_id)==200:
            break
    if int(img_id)==100:
        print("Images have been Captured.")
    captureDevice.release()
    cv2.destroyAllWindows() 
    print("Images have been collected.")
    #train("stay")

def RecogonizeFace():
    #model = load_model("facerecognition.h5")
    #model = keras.models.load_model("Model/facerecognition.h5")
    #loss, acc = model.evaluate(test_images, test_labels)
    #print("Restoed model, accuracy: {:5.3f}%",format(100*acc))
    model = train("send")
    print("Model Successfully Loaded")
    print("Starting to Load Labels....")
    labels = {}
    with open("labels.pickle","rb") as f:
        labels = pickle.load(f)
        labels = {v:k for k,v in labels.items()}
    print(labels)
    face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_alt2.xml")
    captureDevice = cv2.VideoCapture(0)
    print("Hey")
    while(True):
        ret, frame = captureDevice.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            print(x, y, w, h)
            #Draw the Box
            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
            cropped_face = gray[y:y+h,x:x+w]
            if cropped_face is not None:
                face = cv2.resize(cropped_face, (200,200))
                img_data = np.array(face)
                img_data = img_data.reshape(200,200,1)
                model_out = model.predict([img_data])[0]
                #print(model_out)
                print(labels[np.argmax(model_out)])
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 0, 0)
                stroke = 2
                cv2.putText(frame,labels[np.argmax(model_out)],(x,y),font,1,color,stroke,cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if (cv2.waitKey(20) & 0xFF == ord('q')):
            break

def main():
    print("Program Has Started.")
    while(True):
        c = int(input("1 - RecogonizeFace 2 - RegisterFace: "))
        if c==1:
            RecogonizeFace()
        elif c==2:
            RegisterNewFace()
        else:
            print("Invalid Input")

if __name__ == "__main__":
    main()