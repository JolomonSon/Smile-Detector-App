import cv2
from random import randrange

video = cv2.VideoCapture(0)
faceClassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    readSuccessful, frame = video.read()
    if readSuccessful:
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    trainedCoordinates = faceClassifier.detectMultiScale(grayFrame)
    trainedFrames = [cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256),randrange(256))) for (x,y,w,h) in trainedCoordinates]
    print(trainedCoordinates)
    cv2.imshow('Smiley Face', frame)
    cv2.waitKey(1)

video.release()
print('Code Completed')