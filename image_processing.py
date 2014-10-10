#!usr/bin/python

'''
Created 31, July 2014

@author: iHub Research

'''

# load required libraries
import urllib
import shutil
import os
import time
import cv2
import links
import sys
import numpy as np 
from PIL import Image, ImageMath
from os.path import isfile, join
from scipy.spatial import distance

''' --------------------------------- Access Cameras ------------------------------'''

# create Route class
class route(object):

    # initialize class
    def __init__(self,name):
        global way 
        way = links.cameras
        if name in way:
            self.name = name
            self.path = os.getcwd()+'/'+self.name
            self.bb=None
        else:
            sys.exit()

    # set working directory
    def set_dir(self):
        global path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.chdir(self.path)

        
    # capture images
    def capture_images(self):
        for i in 'abc':
            if self.name in way:
                    urllib.urlretrieve(way[self.name],'img_'+i+'.jpg')
                    time.sleep(6)

    # load images
    def load(self):
        files = os.listdir(self.path)
        a = dict()
        b = dict()
        k = 0

        while k <= len(files):
            for names in files:
                if names != '.DS_Store':
                    a[names] = Image.open(names).convert('L')
                    a[names].load()
                    b[names] = np.asarray(a[names])     
            k +=1

        # delete image folder
        shutil.rmtree(os.getcwd())
            
        return b

    # differential imaging
    def diffImg(self,img1,img2,img3):

        # calculate absolute difference
        d1 = cv2.absdiff(img1,img2)
        d2 = cv2.absdiff(img2,img3)
        bit = cv2.bitwise_and(d1,d2)
        ret,thresh = cv2.threshold(bit,35,255,cv2.THRESH_BINARY)

        #get number of different pixels
        moving = list()
        for cell in thresh.flat:
            if cell == 255:
                move = 'True'
                moving.append(move)
            pixie = len(moving)

        return pixie
 
    # calculate optical flow of points on images
    def opticalFlow(self,img1,img2,img3):

        #set variables
        lk_params = dict(winSize = (10,10),
                        maxLevel = 5,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

        features_param = dict( maxCorners = 3000,
                                qualityLevel = 0.5,
                                minDistance = 3,
                                blockSize = 3)

        # feature extraction of points to track 
        pt = cv2.goodFeaturesToTrack(img1,**features_param)
        p0 =np.float32(pt).reshape(-1,1,2)

        # calaculate average movement
        dist = list()
        for loop in p0: 
            p1,st,err =cv2.calcOpticalFlowPyrLK(img1, img2,loop,
                                                None,**lk_params)
      
            p0r,st,err =cv2.calcOpticalFlowPyrLK(img2,img1,p1,
                                            None,**lk_params)

            if abs(loop-p0r).reshape(-1, 2).max(-1) < 1:
                dst = distance.euclidean(loop,p0r)
                dist.append(dst)
        
        return round(max(dist)*10,2)
    
       
