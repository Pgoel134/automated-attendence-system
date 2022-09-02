import cv2
import numpy as np
import face_recognition

imgactual=face_recognition.load_image_file('imagesbasic/harshtomar.jpg')
imgactual = cv2.cvtColor(imgactual, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imagesbasic/PUNJGOELTEST.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgactual)[0]
encodeimg=face_recognition.face_encodings(imgactual)[0]
cv2.rectangle(imgactual,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
print(faceloc)
#print(encodeimg)

faceloctest=face_recognition.face_locations(imgTest)[0]
encodeimgtest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

print(faceloctest)
#print(encodeimgtest)

results=face_recognition.compare_faces([encodeimg],encodeimgtest)
facedis=face_recognition.face_distance([encodeimg],encodeimgtest)
print(results,facedis)
cv2.putText(imgactual,f'{results}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('harshtomar',imgactual)
cv2.imshow('PUNJGOELTEST',imgTest)
cv2.waitKey(0)
