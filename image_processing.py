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
import itertools
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

    # motion detection
    def motion(self,img1,img2,img3):

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

        points = list()
        vally = list()
        for p,val in enumerate(p0.flat):
            points.append(p)
            vally.append(val)
        features = zip(points,vally)
          
        
        # lucas-kanade optical flow (get next points)
        p1,st,err =cv2.calcOpticalFlowPyrLK(img1, img2,p0,
                                            None,**lk_params)
      
        # forward-backward error detection
        p0r,st,err =cv2.calcOpticalFlowPyrLK(img2,img1,p1,
                                            None,**lk_params)
       
        # get correctly predicted points via absolute difference
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

        # Initialize a list to hold new keypoints
        new_keypoints = list()

        # cycle through all current and new keypoints and only keep
        # those that satisfy the "good" condition above
        for (x, y), good_flag in zip(p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            new_keypoints.append((x,y))

        # get tracked points
        meat =list()
        iko = list()
        good_points = np.int32(new_keypoints)

        for me,it in enumerate(good_points.flat):
            meat.append(me)
            iko.append(it)
        new_features = zip(itertools.repeat(it),itertools.repeat(me))

        diff = np.intersect1d(good_points,p0)

        # calculate euclidean distance
        dist = list()
        for pos1 in diff.flat:
            for pos2 in p0.flat:
                track = sp.euclidean(pos1,pos2)
                dist.append(track)

        # get differential image
        d1 = cv2.absdiff(img1,img2)
        d2 = cv2.absdiff(img2,img3)
        bit = cv2.bitwise_and(d1,d2)
        ret,thresh = cv2.threshold(bit,35,255,cv2.THRESH_BINARY)

        moving = list()
      
        for cell in thresh.flat:
            if cell == 255:
                move = 'True'
                moving.append(move)
            pixie = len(moving)

        return features,new_features

        
         
      
