import cv2
from random import randrange

video = cv2.VideoCapture(0)
#Face and Smile Classifier
faceClassifier = cv2.CascadeClassifier('frontalFaceDetector.xml')
smileClassifier = cv2.CascadeClassifier('smileDetector.xml')
while True:
    readSuccessful, frame = video.read()
    if not readSuccessful:
        break
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    trainedFaceCoordinates = faceClassifier.detectMultiScale(grayFrame, scaleFactor=1.7, minNeighbors=20)
    trainedSmileCoordinates = smileClassifier.detectMultiScale(grayFrame, scaleFactor=1.7, minNeighbors=20)
    trainedFaceFrame = [cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0),4) for (x,y,w,h) in trainedFaceCoordinates]
    trainedSmileFrame = [cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255)) for (x, y, w, h) in trainedSmileCoordinates]
    print('Face Coordinate',trainedFaceCoordinates)
    print('Smile Coordinate',trainedSmileCoordinates)
    cv2.imshow('Smiley Face', frame)
    cv2.waitKey(1)


video.release()
cv2.destroyAllWindows()
print('Code Completed')