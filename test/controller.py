import cv2            # 引入 OpenCV 的模組，製作擷取攝影機影像之功能
import sys, time      # 引入 sys 跟 time 模組
import numpy as np    # 引入 numpy 來處理讀取到得影像矩陣
import face_recognition
import os
import pickle
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import pyqtSlot, QTimer, QDate, Qt
from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow
import datetime
import shutil
import dlib
import ipfshttpclient
from threading import Timer
import time
import json
from web3 import Web3
import requests

class MainWindow_controller(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow_controller, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.viewData.setScaledContents(True) 
        self.viewData_2.setScaledContents(True) 
        self.str1 = ""
        self.Ny = 0
        self.Nx = 0
        self.n = 0  
        self.i = 1          
        # 連接按鍵
        self.camBtn_open.clicked.connect(self.show_camera)  # 槽功能：開啟攝影機
        self.camBtn_stop.clicked.connect(self.stopCam)  # 槽功能：暫停讀取影像
        self.trainbtn.clicked.connect(self.trainSave)
        self.uploadbtn.clicked.connect(self.upload)
        self.host = "http://127.0.0.1:5001"
        self.url_download = self.host + "/api/v0/cat"
        self.pushButton_2.clicked.connect(self.cat_hash)

        
        
    def stopCam(self):
        """ 凍結攝影機的影像 """
        self.bool = 0
        self.setup_control()
        # 按鈕的狀態：啟動 ON、暫停 OFF、視窗大小 OFF
        self.camBtn_open.setEnabled(True)
        self.camBtn_stop.setEnabled(False)          
        
    def show_camera(self):
        self.bool = 1
        window_title = "Camera"
        self.camBtn_open.setEnabled(False)
        self.camBtn_stop.setEnabled(True) 
        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        self.video_capture = cv2.VideoCapture(0)
        if self.video_capture.isOpened():
            self.Encodings=[]
            self.Names=[]

            with open('train.pkl','rb') as f:
                self.Names=pickle.load(f)
                self.Encodings=pickle.load(f)
            self.font=cv2.FONT_HERSHEY_SIMPLEX
            ganache_url = "http://127.0.0.1:8545/"
            self.web3 = Web3(Web3.HTTPProvider(ganache_url))
            self.web3.eth.defaultAccount = self.web3.eth.accounts[0]
            abi = json.loads('[{"constant":false,"inputs":[{"name":"x","type":"string"}],"name":"set","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"get","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"ipfsHash","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"}]')
            address = self.web3.toChecksumAddress("0xf2790ff955Eb85F8C318968f2b0775C2b00ebdC4")
            self.contract = self.web3.eth.contract(address=address, abi=abi)
            
            while True:
                ret_val, self.frame = self.video_capture.read()
                    # Check to see if the user closed the window
                    # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                    # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                    #if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                        #cv2.imshow(window_title, frame)
            
                try:
                    image = self.face_recognition(self.frame, self.Encodings, self.Names ,self.font)
                except Exception as e:
                    print(e)
                
                self.Ny, self.Nx, _ = image.shape  # 取得影像尺寸
                #print(self.Ny)
                #print(self.Nx)
                self.qimg = QtGui.QImage(image.data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)
                self.viewData.setPixmap(QtGui.QPixmap.fromImage(self.qimg))

                keyCode = cv2.waitKey(10) & 0xFF
                    # Stop the program on the ESC key or 'q'
                    #if keyCode == 27 or keyCode == ord('q'):
                     #   break
                if self.bool == 0:
                    break
                    video_capture.release()
            #finally:
                #video_capture.release()
        else:
            print("Error: Unable to open camera")
    
    
    def setup_control(self):
        # TODO
        self.img_path = 'cat_small.jpg'
        self.display_img()

    def display_img(self):
        self.img = cv2.imread(self.img_path)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QtGui.QImage(self.img, width, height, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.viewData.setPixmap(QtGui.QPixmap.fromImage(self.qimg))
    
    def trainSave(self):
        Encodings=[]
        Names=[]

        image_dir='C:/Users/jimmy/Desktop/test/faceRecognmizer/demoImages-master/known'
        for root, dirs, files in os.walk(image_dir):
            print(files)
            for file in files:
                path=os.path.join(root,file)
                print(path)
                name=os.path.splitext(file)[0]
                print(name)
                person=face_recognition.load_image_file(path)
                encoding=face_recognition.face_encodings(person)[0]
                Encodings.append(encoding)
                Names.append(name)
        print(Names)
        self.label_6.setText('訓練完成，若有再上傳照片，請重新訓練')

        with open('train.pkl','wb') as f:
            pickle.dump(Names,f)
            pickle.dump(Encodings,f)
    
    def face_recognition(self, img, Encodings, Names, font):
        imgSmall=cv2.resize(img,(0,0),fx=0.5,fy=0.5)
        imgRGB=cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)
        facePositions=face_recognition.face_locations(imgRGB)#, number_of_times_to_upsample="0")
        allEncodings=face_recognition.face_encodings(imgRGB,facePositions)
        for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
            name='Unkown Person'
            #print(top,right,bottom,left)
            print(self.n)
            matches=face_recognition.compare_faces(Encodings,face_encoding)
            if True in matches:
                first_match_index=matches.index(True)
                name=Names[first_match_index]
            else:
                #client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001') 
                #print("IPFS client id", client.id)
                #res = client.add('<your student id>.txt')
                # 我的測試程式這一行是res = client.add('9200763.txt')
                #print(res)   
                if self.n > 5:
                    localtime = time.localtime() 
                    x,y,_=imgSmall.shape
                    result = time.strftime( "%Y-%m-%d %I :%M:%S %p" , localtime)
                    cv2.putText(imgSmall,result,(0,25),font,1,(0,0,255),1)
                    top=top*1
                    right=right*1
                    bottom=bottom*1
                    left=left*1
                    cv2.rectangle(imgSmall,(left,top),(right, bottom),(0,255,0),1)
                    #cv2.imshow(imgRGB)
                    cv2.putText(imgSmall,name,(left,top-6),font,1,(0,0,255),1)
                    path1='C:/Users/jimmy/Desktop/test/unknownperson/unknown%d.jpg'%self.i
                    cv2.imwrite(path1,imgSmall)
                    client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001') 
                    #print("IPFS client id", client.id)
                    res = client.add(path1)
                    #print(res['Hash'])
                    hash1=self.contract.functions.set(res['Hash']).transact()
                    self.web3.eth.wait_for_transaction_receipt(hash1)
                    hash2=self.contract.functions.get().call()
                    print(hash2)   
                    self.textBrowser_2.append('unknown%d.jpg'%self.i)
                    self.textBrowser_2.append(result)
                    self.textBrowser_2.append(hash2)
                    self.textBrowser_2.ensureCursorVisible()
                    self.i = self.i +1
                    self.n = 0
            top=top*1
            right=right*1
            bottom=bottom*1
            left=left*1
            cv2.rectangle(imgRGB,(left,top),(right, bottom),(0,255,0),2)
            #cv2.imshow(imgRGB)
            cv2.putText(imgRGB,name,(left,top-6),font,.5,(0,0,255),2)
            self.n = self.n +1
        return imgRGB

    def upload(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./",
                  "PNG files (*.png);;JPG files (*.jpg);;JPEG files (*.jpeg")                 # start path
        f =os.path.splitext(filename, )
        bk_file = f[0]+"_bk_"+self.mytime()+f[1]
        #backup = os.path.join('C:/Users\Administrator/Desktop/test/faceRecognmizer/demoImages-master/known',bk_file)
        shutil.copy(filename,'C:/Users/jimmy/Desktop/test/faceRecognmizer/demoImages-master/known')
        s = time.time()
        self.label_5.setText('上傳成功')
        
    def mytime(self):
        now = datetime.date.today()
        return str(now)

    def setup_control1(self):
        # TODO
        self.img_path1 = 'C:/Users/jimmy/Desktop/test/download/test.jpeg'
        self.display_img1()

    def display_img1(self):
        self.img1 = cv2.imread(self.img_path1)
        height1, width1, channel1 = self.img1.shape
        bytesPerline1 = 3*width1
        self.qimg1 = QtGui.QImage(self.img1, width1, height1, bytesPerline1, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.viewData_2.setPixmap(QtGui.QPixmap.fromImage(self.qimg1))  
        
    def cat_hash(self):
        """
        讀取文件內容
        :param hash_code:
        :return:
        """
        self.str1 = self.textEdit.toPlainText()
        params = {
            'arg': self.str1
        }
        response = requests.post(self.url_download, params=params)  
        #print(response.content)
        with open('C:/Users/jimmy/Desktop/test/download/test.jpeg', mode='wb') as f:
                f.write(response.content)
        self.setup_control1()
                        
        #self.plx = QtGui.QImage(response.text)
        #self.label.setPixmap(QtGui.QPixmap.fromImage(self.plx))
        
    