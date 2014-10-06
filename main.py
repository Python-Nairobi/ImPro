'''
Created 13 August, 2014
@author : iHub Research

'''

# load required libraries
import image_processing
import numpy as np
from flask import Flask
import links
import json
import scipy as sp

# create flask web server 
#app = Flask(__name__)

# create HTTP endpoint
#@app.route('/ImPro/<road>')

# main function
# def get_route(road):
# initialize route class
road = 'sarit'
traffic = image_processing.route(road)

# setup working directory
traffic.set_dir()

# get image from traffic camera
traffic.capture_images()

# load image stack
x = traffic.load()

# differential imaging
y =  traffic.diffImg(x['img_a.jpg'],x['img_b.jpg'],x['img_c.jpg'])

# calculate optical flow
z = traffic.opticalFlow(x['img_a.jpg'],x['img_b.jpg'],x['img_c.jpg'])
print (z)

#if __name__ == "__main__":
#app.run()
