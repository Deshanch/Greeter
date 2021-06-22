import os #this used to read file like operations 
import cv2 #for computer vision
import numpy as np #for mathematical operations 
from face_recognition import *
from keras.models import model_from_json #to save and load keras models 
from keras.preprocessing import image

#load model
model = model_from_json(open("fer.json", "r").read())#load the model
#load weights
model.load_weights('fer.h5')#to load the weights from the saved file

#this cascade  is used to detect faces inside an image  --> haarcascade_frontalface_default.xml
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap=cv2.VideoCapture(0)#to call the first camera using object and if we use the second camera we have to pass the 1 to the argument
        
while True:
    try:
        ret,test_img=cap.read()# captures frame and returns boolean value and captured image
        if not ret:
            continue #if no image was captured then continue the getting of images and otherwise process
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)#converting the rgb to gray colour image

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 10)


        for (x,y,w,h) in faces_detected:

            reco_img = test_img[y-20:y + h+20, x-20:x + w+20]
            cv2.imshow('recognized image', reco_img)
            reco_img = cv2.resize(reco_img, (96, 96))
            reco_img = reco_img[...,::-1]
            name = capture(reco_img) 

            roi_gray = gray_img[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #predicting the emotion
            yhat= model.predict(cropped_img)
            cv2.putText(test_img,name+"  "+ labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            print("Emotion: "+labels[int(np.argmax(yhat))])

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ',resized_img)

        if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
            break

    except Exception as e:
        print(str(e))
        print("something went wrong")
        continue
    
cap.release()#don’t forget to release the capture
cv2.destroyAllWindows