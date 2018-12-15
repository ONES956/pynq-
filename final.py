# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 19:47:38 2018

@author: hp
"""

#按键0实时、按键1拍照（默认灰色）、按键2（在按下按键1后）将灰度图对应彩图增强展示、按键3退出
import PIL.Image
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
import numpy as np
import cv2
base = BaseOverlay("base.bit")

Mode = VideoMode(1280,1024,8)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode,PIXEL_GRAY)
hdmi_out.start()
out=False
test=False
count=0
shootCount=0
num=0

# camera (input) configuration
frame_in_w = 640
frame_in_h = 480

videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);

# Capture webcam image

def frameHE1(fframe):
    fframe=np.array(fframe) 
    for k in range(1):
        #a=fframe[:,:,k]
        a=fframe
        m,n=np.shape(a)
    #计算图像直方图（每个bins数组的区间值对应一个imhist数组中的强度值）
        imhist,bins = np.histogram(a.flatten(),256,normed=True)
        l=np.zeros(shape=(256,1))
        t=np.mean(imhist)
        j=0
        for i in imhist:           
            if i>t:
                l[j]=t
            else:
                l[j]=i
            j+=1 
#计算累积分布函数
        cdf =l.cumsum()
#累计函数归一化（由0～1变换至0~255）
        cdf = cdf*255/cdf[-1]
#依次对每一个灰度图像素值（强度值）使用cdf进行线性插值，计算其新的强度值
#interp（x，xp，yp） 输入原函数的一系列点（xp，yp），使用线性插值方法模拟函数并计算f（x）
        im2 = np.interp(a.flatten(),bins[:256],cdf)
#将压平的图像数组重新变成二维数组
        im2 = im2.reshape(m,n)
        #fframe[:,:,k]=im2
        fframe=im2
    fframe=np.uint8(fframe)
    return fframe

def frameProcess(frame):
    xframe=np.zeros([480,640,3])
    for i in range(3):
        a=frame[:,:,i]
        Max=a.max()
        Min=a.min()
        a=(a-Min)/(Max-Min)*255/8
        xframe[:,:,i]=a
    xframe=np.uint8(xframe)
    return xframe
def frameProcess1(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=np.array(gray)
    Max=gray.max()
    Min=gray.min()
    gray=(gray-Min)/(Max-Min)*255/8
    gray=np.uint8(gray)
    return gray
def frameHE(fframe):
    fframe=np.array(fframe) 
    for k in range(3):
        a=fframe[:,:,k]
        #a=fframe
        m,n=np.shape(a)
    #计算图像直方图（每个bins数组的区间值对应一个imhist数组中的强度值）
        imhist,bins = np.histogram(a.flatten(),256,normed=True)
        l=np.zeros(shape=(256,1))
        t=np.mean(imhist)
        j=0
        for i in imhist:           
            if i>t:
                l[j]=t
            else:
                l[j]=i
            j+=1 
#计算累积分布函数
        cdf = l.cumsum()
#累计函数归一化（由0～1变换至0~255）
        cdf = cdf*255/cdf[-1]
#依次对每一个灰度图像素值（强度值）使用cdf进行线性插值，计算其新的强度值
#interp（x，xp，yp） 输入原函数的一系列点（xp，yp），使用线性插值方法模拟函数并计算f（x）
        im2 = np.interp(a.flatten(),bins[:256],cdf)
#将压平的图像数组重新变成二维数组
        im2 = im2.reshape(m,n)
        fframe[:,:,k]=im2
    fframe=np.uint8(fframe)
    return fframe


outframe = hdmi_out.newframe()
while(videoIn.isOpened()):
    if (base.buttons[3].read()==1)or(out): 
        out=False
        break
    ret, mframe_vga = videoIn.read()
    if ret:
        mfframe=cv2.cvtColor(mframe_vga,cv2.COLOR_BGR2GRAY)
        mfframe1=frameHE1(mfframe) 
        if count==20:
            count=0
            np_frame1 = mfframe1
            face_cascade = cv2.CascadeClassifier(
             '/home/xilinx/jupyter_notebooks/base/video/data/'
             'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(np_frame1, 1.3, 5)
            gray=np_frame1
            for (x,y,w,h) in faces:
                cv2.rectangle(np_frame1,(x,y),(x+w,y+h),(0,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = np_frame1[y:y+h, x:x+w]
            if len(faces)!=0:
                test=True
                num+=1
                for j in range(8):
                     img=PIL.Image.fromarray(mframe_vga[:,:,[2,1,0]])
                     img.save("/home/xilinx/jupyter_notebooks/result/%d%d.jpg"%num,%j)
                     ret, mframe_vga = videoIn.read()
                     if (ret==False):
                          out=True
                          break                     
        count+=1
    if (test):      
        outframe[0:480,640:1280] = np_frame1[0:480,0:640]
        test=False
    if (ret):      
        outframe [0:480,0:640]= mfframe[0:480,0:640]
        outframe[480:960,640:1280]= mfframe1[0:480,0:640]
    hdmi_out.writeframe(outframe)
    if (base.buttons[1].read()==1):             
        if(videoIn.isOpened()):
            ret1, frame_vga1= videoIn.read()
            ret2, frame_vga2= videoIn.read()
            ret3, frame_vga3= videoIn.read()
            ret4, frame_vga4= videoIn.read()
            ret5, frame_vga5= videoIn.read()
            ret6, frame_vga6= videoIn.read()
            ret7, frame_vga7= videoIn.read()
            ret8, frame_vga8= videoIn.read()
            if  (ret8):
                fframe_vga=np.zeros([480,640])
                fframe1=frameProcess1(frame_vga1)
                fframe2=frameProcess1(frame_vga2)
                fframe3=frameProcess1(frame_vga3)
                fframe4=frameProcess1(frame_vga4)
                fframe5=frameProcess1(frame_vga5)
                fframe6=frameProcess1(frame_vga6)
                fframe7=frameProcess1(frame_vga7)
                fframe8=frameProcess1(frame_vga8)
                fframe_vga=fframe1+ fframe2+fframe3+fframe4+fframe5+fframe6+fframe7+fframe8
            xfframe=frameHE1(fframe_vga)
            img_median1 = cv2.medianBlur(xfframe, 3)
            fframe1=cv2.cvtColor(frame_vga1,cv2.COLOR_BGR2GRAY)
            np_frame = fframe1
            face_cascade = cv2.CascadeClassifier(
             '/home/xilinx/jupyter_notebooks/base/video/data/'
             'haarcascade_frontalface_default.xml')
#                eye_cascade = cv2.CascadeClassifier(
#                 '/home/xilinx/jupyter_notebooks/base/video/data/'
#                'haarcascade_eye.xml')
            faces = face_cascade.detectMultiScale(np_frame, 1.3, 5)

            for (x,y,w,h) in faces:
                cv2.rectangle(np_frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = np_frame[y:y+h, x:x+w]
#                    eyes = eye_cascade.detectMultiScale(roi_gray)
#                    for (ex,ey,ew,eh) in eyes:
#                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            np_frame = fframe_vga
            face_cascade = cv2.CascadeClassifier(
             '/home/xilinx/jupyter_notebooks/base/video/data/'
             'haarcascade_frontalface_default.xml')
#                eye_cascade = cv2.CascadeClassifier(
#                '/home/xilinx/jupyter_notebooks/base/video/data/'
#                'haarcascade_eye.xml')
            gray=np_frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(np_frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = np_frame[y:y+h, x:x+w]
#                    eyes = eye_cascade.detectMultiScale(roi_gray)
#                    for (ex,ey,ew,eh) in eyes:
#                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 
            np_frame = xfframe
            face_cascade = cv2.CascadeClassifier(
            '/home/xilinx/jupyter_notebooks/base/video/data/'
            'haarcascade_frontalface_default.xml')
#                eye_cascade = cv2.CascadeClassifier(
#                '/home/xilinx/jupyter_notebooks/base/video/data/'
#                'haarcascade_eye.xml')
            gray=np_frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(np_frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = np_frame[y:y+h, x:x+w]
#                    eyes = eye_cascade.detectMultiScale(roi_gray)
#                    for (ex,ey,ew,eh) in eyes:
#                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            np_frame = img_median1
            face_cascade = cv2.CascadeClassifier(
            '/home/xilinx/jupyter_notebooks/base/video/data/'
            'haarcascade_frontalface_default.xml')
#                eye_cascade = cv2.CascadeClassifier(
#                '/home/xilinx/jupyter_notebooks/base/video/data/'
#                'haarcascade_eye.xml')
            gray=np_frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(np_frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = np_frame[y:y+h, x:x+w]
#                    eyes = eye_cascade.detectMultiScale(roi_gray)
#                    for (ex,ey,ew,eh) in eyes:
#                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            outframe = hdmi_out.newframe()
            outframe[0:480,0:640] = fframe1[0:480,0:640]
            outframe[0:480,640:1280] = fframe_vga[0:480,0:640]
            outframe[480:960,0:640] = xfframe[0:480,0:640]
            outframe[480:960,640:1280] =img_median1[0:480,0:640]
            hdmi_out.writeframe(outframe)
            shoot1=True
            while True:  
                if (base.buttons[0].read()==1):
                    hdmi_out.stop()
                    del hdmi_out
                    base = BaseOverlay("base.bit")
                    # monitor configuration: 640*480 @ 60Hz
                    Mode = VideoMode(1280,1024,8)
                    hdmi_out = base.video.hdmi_out
                    hdmi_out.configure(Mode,PIXEL_GRAY)
                    hdmi_out.start()
                    videoIn.release()
                    videoIn = cv2.VideoCapture(0)
                    videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
                    videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
                    outframe = hdmi_out.newframe()
                    break
                if (base.buttons[3].read()==1):
                    out=True
                    break
                if (base.buttons[1].read()==1):         #cv2.waitKey(0)&      
                    hdmi_out.stop()
                    del hdmi_out
                    base = BaseOverlay("base.bit")
                    Mode = VideoMode(1280,1024,8)
                    hdmi_out = base.video.hdmi_out
                    hdmi_out.configure(Mode,PIXEL_GRAY)
                    hdmi_out.start()
                    outframe = hdmi_out.newframe()
                    outframe[0:480,0:640] = fframe1[0:480,0:640]
                    outframe[0:480,640:1280] = fframe_vga[0:480,0:640]
                    outframe[480:960,0:640] = xfframe[0:480,0:640]
                    outframe[480:960,640:1280] =img_median1[0:480,0:640]
                    hdmi_out.writeframe(outframe)
                if (base.buttons[2].read()==1):          
                    frame_vga=np.zeros([480,640,3])
                    frame1=frameProcess(frame_vga1)
                    frame2=frameProcess(frame_vga2)
                    frame3=frameProcess(frame_vga3)
                    frame4=frameProcess(frame_vga4)
                    frame5=frameProcess(frame_vga5)
                    frame6=frameProcess(frame_vga6)
                    frame7=frameProcess(frame_vga7)
                    frame8=frameProcess(frame_vga8)
                    frame_vga=frame1+ frame2+frame3+frame4+frame5+frame6+frame7+frame8
                    xframe=frameHE(frame_vga)
                    img_median = cv2.medianBlur(xframe, 3)
                    np_frame = frame_vga1
                    face_cascade = cv2.CascadeClassifier(
                     '/home/xilinx/jupyter_notebooks/base/video/data/'
                     'haarcascade_frontalface_default.xml')
    #                eye_cascade = cv2.CascadeClassifier(
    #                 '/home/xilinx/jupyter_notebooks/base/video/data/'
    #                 'haarcascade_eye.xml')
                    gray = cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY)
                    #gray = np_frame
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    for (x,y,w,h) in faces:
                        cv2.rectangle(np_frame,(x,y),(x+w,y+h),(255,0,0),2)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = np_frame[y:y+h, x:x+w]
    #                    eyes = eye_cascade.detectMultiScale(roi_gray)
    #                    for (ex,ey,ew,eh) in eyes:
    #                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    np_frame = frame_vga
                    face_cascade = cv2.CascadeClassifier(
                     '/home/xilinx/jupyter_notebooks/base/video/data/'
                     'haarcascade_frontalface_default.xml')
    #                eye_cascade = cv2.CascadeClassifier(
    #                '/home/xilinx/jupyter_notebooks/base/video/data/'
    #                'haarcascade_eye.xml')
                    gray = cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY)
                    #gray = np_frame
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(np_frame,(x,y),(x+w,y+h),(255,0,0),2)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = np_frame[y:y+h, x:x+w]
    #                    eyes = eye_cascade.detectMultiScale(roi_gray)
    #                    for (ex,ey,ew,eh) in eyes:
    #                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 
                    np_frame = xframe
                    face_cascade = cv2.CascadeClassifier(
                    '/home/xilinx/jupyter_notebooks/base/video/data/'
                    'haarcascade_frontalface_default.xml')
    #                eye_cascade = cv2.CascadeClassifier(
    #                '/home/xilinx/jupyter_notebooks/base/video/data/'
    #                'haarcascade_eye.xml')
                    gray = cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY)
                #gray = np_frame
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(np_frame,(x,y),(x+w,y+h),(255,0,0),2)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = np_frame[y:y+h, x:x+w]
    #                    eyes = eye_cascade.detectMultiScale(roi_gray)
    #                    for (ex,ey,ew,eh) in eyes:
    #                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    np_frame = img_median
                    face_cascade = cv2.CascadeClassifier(
                    '/home/xilinx/jupyter_notebooks/base/video/data/'
                    'haarcascade_frontalface_default.xml')
    #                eye_cascade = cv2.CascadeClassifier(
    #                '/home/xilinx/jupyter_notebooks/base/video/data/'
    #                'haarcascade_eye.xml')
                    gray = cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY)
                #gray = np_frame
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(np_frame,(x,y),(x+w,y+h),(255,0,0),2)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = np_frame[y:y+h, x:x+w]
    #                    eyes = eye_cascade.detectMultiScale(roi_gray)
     #                   for (ex,ey,ew,eh) in eyes:
    #                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    hdmi_out.stop()
                    del hdmi_out
                    base = BaseOverlay("base.bit")
                    # monitor configuration: 640*480 @ 60Hz
                    Mode = VideoMode(1280,1024,24)
                    hdmi_out = base.video.hdmi_out
                    hdmi_out.configure(Mode,PIXEL_BGR)
                    hdmi_out.start()
                    outframe = hdmi_out.newframe()
                    outframe[0:480,0:640,:] = frame_vga1[0:480,0:640,:]
                    outframe[0:480,640:1280,:] = frame_vga[0:480,0:640,:]
                    outframe[480:960,0:640,:] = xframe[0:480,0:640,:]
                    outframe[480:960,640:1280,:] =img_median[0:480,0:640,:]
                    hdmi_out.writeframe(outframe)
videoIn.release()
hdmi_out.stop()
del hdmi_out  


