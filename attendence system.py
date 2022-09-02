import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='imagesattendence'
images=[]
classnames=[]
mylist=os.listdir(path)
#print(mylist)
for cls in mylist:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classnames.append(os.path.splitext(cls)[0])
print(classnames)

def findencodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodeimg=face_recognition.face_encodings(img)[0]
        encodelist.append(encodeimg)
    return encodelist

def markAttendence(name):
    with open('attendence sheet.csv','r+') as f:
        mydatalist = f.readlines()
        namelist =[]
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


              
encodelistknown = findencodings(images)
print('encoding complete')

cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    
    facecurrent=face_recognition.face_locations(imgS)
    encodecurrent=face_recognition.face_encodings(imgS,facecurrent)

    for encodeFace,faceLoc in zip(encodecurrent,facecurrent):
        matches = face_recognition.compare_faces(encodelistknown,encodeFace)
        facedis = face_recognition.face_distance(encodelistknown,encodeFace)
        #print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y1-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)

            
    cv2.imshow('webcam',img)
    cv2.waitKey(1)

    

    

#faceloc=face_recognition.face_locations(imgactual)[0]
#encodeimg=face_recognition.face_encodings(imgactual)[0]
#cv2.rectangle(imgactual,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
#print(faceloc)
#print(encodeimg)

#faceloctest=face_recognition.face_locations(imgTest)[0]
#encodeimgtest=face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

#print(faceloctest)
#print(encodeimgtest)

#results=face_recognition.compare_faces([encodeimg],encodeimgtest)
#facedis=face_recognition.face_distance([encodeimg],encodeimgtest)
#print(results,facedis)



