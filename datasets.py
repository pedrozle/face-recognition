import cv2
import numpy as np
from keras.applications import VGG16
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,  Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None
    
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    
    return cropped_face

img_rows, img_cols = 224, 224

vgg = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

for layer in vgg.layers:
    layer.trainable = False


def Fc(bot_model, num_classes):
    top_model = bot_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model
    

num_classes = 3

FC_Head = Fc(vgg, num_classes)

model = Model(inputs= vgg.input, outputs= FC_Head)

# https://medium.com/@mansi.dadheech22/face-recognition-using-vgg-b51f44650e26 link para terminar depois