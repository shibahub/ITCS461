from importlib.resources import path
import cv2
import numpy
import face_recognition
import os

path = 'Pic'
image=[]
Name=[]
Listuser= os.listdir(path)
print(Listuser)
for idenUser in Listuser:
    inputImg=cv2.imread(f'{path}/{idenUser}')
    image.append(inputImg)
    Name.append(os.path.splitext(idenUser)[0])
print(Name)


def encodeImg(image):
    encodeList=[]
    for img in image:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.encode(img)[0]
        encodeList.append(encode)
    return encodeList



encodeKnownuser= encodeImg(image)
print(len(encodeKnownuser))