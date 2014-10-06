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

        # convert corner points to floating-point
        p0 =np.float32(pt).reshape(-1,1,2)

        # generate index for corner points
        points = list()
        vally = list()
        for p,val in enumerate(p0):
            points.append(p)
            vally.append(val)
        features = zip(points,vally)

        column = 0
        column_list = []

        for j in xrange(len(vally)):
            column_list += [vally[j][column]]

        # get next points using lucas-kanade optical flow 
        p1,st,err =cv2.calcOpticalFlowPyrLK(img1, img2,p0,
                                            None,**lk_params)
      
        # forward-backward error detection
        p0r,st,err =cv2.calcOpticalFlowPyrLK(img2,img1,p1,
                                            None,**lk_params)
       
        # get correctly predicted points via absolute difference
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

        # cycle through all current and new keypoints and only keep
        # those that satisfy the "good" condition above

        # Initialize a list to hold new keypoints
        new_keypoints = list()
        new_indices = list()

        for (x, y), good_flag,ind in zip(p1.reshape(-1, 2), good,enumerate(good)):
            if not good_flag:
                continue
            new_keypoints.append((x,y))
            new_indices.append(ind)

        # generate index for new points
        meat =list()
        iko = list()
        good_points = np.int32(new_keypoints)

        for me,it in enumerate(good_points):
            meat.append(me)
            iko.append(it)
        new_features = zip(meat,iko)

        col = 0
        colList = []

        for i in xrange(len(new_indices)):
            colList += [new_indices[i][col]]

        return p0,column_list,new_keypoints,colList

    
       
