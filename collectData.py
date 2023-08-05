import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap=cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 500
folder = 'Data/C'
counter=0
while True:
    success,img = cap.read()
    hands, img =detector.findHands(img)
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
        else:
            k = imgSize / w
            CalcH = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize,CalcH))
            imgResizeShape = imgResize.shape
            GapH = math.ceil((imgSize - CalcH) / 2)
            imgWhite[GapH:CalcH + GapH,:] = imgResize
        cv2.imshow("Cropped Image", imgCrop)
        cv2.imshow("Image WHite", imgWhite)
    cv2.imshow("Image",img)
    key=cv2.waitKey(1)
    if key == ord('s'):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)