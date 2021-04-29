import cv2
from random import randrange
# Real Time Image(frame)
video  = cv2.VideoCapture(0)
# Face and Smile Classifiers
faceClassifier = cv2.CascadeClassifier('frontalFaceDetector.xml')
smileClassifier = cv2.CascadeClassifier('smileDetector.xml')
while True:
    readSuccessful, frame = video.read()
    if not readSuccessful:
        break
    # Converting Image to grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # Classifiers Coordinates
    faceCoordinates = faceClassifier.detectMultiScale(grayFrame)
    smileCoordinates = smileClassifier.detectMultiScale(grayFrame, scaleFactor=1.7, minNeighbors=20)
    print('Face Coordinates - ',faceCoordinates,'\nSmile Coordinates - ',smileCoordinates)
    # Rectangles on the face
    #for (x,y,w,h) in faceCoordinates:
    faceFrame = [cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256), randrange(256), randrange(256))) for (x,y,w,h) in faceCoordinates]
    #smileFrame = [cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256), randrange(256), randrange(256)),2) for (x,y,w,h) in smileCoordinates]
    for (x,y,w,h) in smileCoordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)),2)
    # Label the face (Smiling or Not Smiling)
        if len(smileCoordinates) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(256,0,256))
        else:
            cv2.putText(frame, 'Not Smiling', (x, y+h+40), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0,256,0))
    cv2.imshow('Webcam Smiley Face Detection',frame)
    key = cv2.waitKey(1)
    # Stop if Q is pressed
    if key==81 or key==113:
        break
#Release the VideoCapture object
print(faceCoordinates)
video.release()