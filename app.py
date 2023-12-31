import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow.keras

cap=cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
#trained_model = load_model('model.h5', compile=False)
classifier = Classifier('Model/keras_model.h5','Model/labels.txt')

offset = 20
imgSize = 500
#folder = 'Data/C'
counter=0
labels = ['A','B','C']
while True:
    success,img = cap.read()
    imgOutput =img.copy()
    hands,img =detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgCropShape = imgCrop.shape
        aspectRatio =h/w
        if aspectRatio >1:
            k = imgSize/h
            CalcW=math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(CalcW,imgSize))
            imgResizeShape = imgResize.shape
            GapW=math.ceil((imgSize-CalcW)/2)
            imgWhite[:,GapW:CalcW+GapW] = imgResize
            prediction,index=classifier.getPrediction(imgWhite,draw=False)
            #print(prediction,index)

        else:
            k = imgSize / w
            CalcH = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize,CalcH))
            imgResizeShape = imgResize.shape
            GapH = math.ceil((imgSize - CalcH) / 2)
            imgWhite[GapH:CalcH + GapH,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)

        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,255,255),4)
        cv2.imshow("Cropped Image", imgCrop)
        cv2.imshow("Image WHite", imgWhite)
    cv2.imshow("Image",imgOutput)
    cv2.waitKey(1)
