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
		for i in 'abcdef':
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
						print 'File Not Loaded'
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
		
		# lucas-kanade optical flow (get next points)
		p1,st,err =cv2.calcOpticalFlowPyrLK(img1, img2,p0,
											None,**lk_params)
      
        # forward-backward error detection
		p0r,st,err =cv2.calcOpticalFlowPyrLK(img2,img1,p1,
											None,**lk_params)
       
       	# absolute difference between image points
		d = abs(p0-p0r).reshape(-1, 2).max(-1)
		good = d < 1

			# feature extraction of points to track 
		pt1 = cv2.goodFeaturesToTrack(img2,**features_param)

		# convert corner points to floating-point
		p01 =np.float32(pt1).reshape(-1,1,2)

        # third image position estimation
		p2,st,err =cv2.calcOpticalFlowPyrLK(img2, img3,p01,
											None,**lk_params)

		return p1,p2
		