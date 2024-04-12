import cv2
import numpy as np
import math
import HandTrackingModule as htm
import time
import os

# Open Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

folderPath = "images"
imageList = os.listdir(folderPath)
print(imageList)

overlayList = []
for imPath in imageList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0
cTime = 0

detector = htm.HandDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    # Read Frame
    success, img = cap.read()
    # Flip Image
    img = cv2.flip(img, 1)
    # Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        fingers = []
        for id in range(0, 5):
            if id !=0:
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if lmList[tipIds[id]][1] < lmList[tipIds[id] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        totalFingers = fingers.count(1)
        print(fingers)
        
        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w] = overlayList[totalFingers] 
        print("IMAGE: ",h, w, c)
        pos = int(w/2)
        cv2.rectangle(img, (0, h), (w, h+60), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (pos - 20, h+55), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)
    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # rhs
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # Display Image
    cv2.imshow("Image", img)
    # Exit
    cv2.waitKey(1)
