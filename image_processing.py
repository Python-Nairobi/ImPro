#!usr/bin/python

'''
Created 31, July 2014

@author: iHub Research

'''

# load required libraries
import urllib
import os
import time
import cv2
import links
import numpy as np 
import scipy.spatial.distance as sp 
from PIL import Image, ImageMath
from os.path import isfile, join

''' --------------------------------- Access Cameras ------------------------------'''

# create Route class
class Route(object):

    # initialize 
    def __init__(self,name):
        self.name = name
        self.path = os.getcwd()+'/'+self.name
        self.bb=None

    # set working directory
    def set_dir(self):
        global path
        #path = os.getcwd()+'/'+self.name
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        return os.chdir(self.path)

    # capture images
    def images(self):
        for i in 'abc':
            way = links.links
            if self.name in way:
                urllib.urlretrieve(way[self.name],'img_'+i+'.jpg')
                time.sleep(6)
            else:
                print 'No Camera for Location'

    # load downloaded images
    def load(self):
        # set variables  
        files = os.listdir(self.path)
        a = dict()
        b = dict()
        k = 0

        # load captured images
        while k <= len(files):
            for names in files:
                if names != '.DS_Store':
                    try:
                        a[names] = Image.open(names).convert('L')
                        a[names].load()
                        b[names] = np.asarray(a[names])
                    except IOError:
                        camera_status = 'Camera Busy'
                        print camera_status
                    except KeyError:
                        camera_status = 'Camera Busy'
                        print camera_status

                k +=1
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

        return p0[0],new_keypoints[0]

    # calculate euclidean distance between pixels
    def distance(self,a,b):

        # euclidean distance between two points
        dist = sp.euclidean(a,b)

        return dist
       

        
         
      
