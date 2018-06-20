import cv2
import numpy as np


facedec = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

idq = input('enter the user id')
samplenum=0

while True:
    suc,frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = facedec.detectMultiScale(gray,1.3,5)

    for x,y,w,h in faces:
        samplenum = samplenum + 1
        name = "dataset/user."+str(idq)+"."+str(samplenum)+".jpg"
        cv2.imwrite(name,gray[y:y+h, x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(50)


    cv2.imshow("faces",frame)
    if samplenum>35:
        break

cam.release()
cv2.destroyAllWindows()

