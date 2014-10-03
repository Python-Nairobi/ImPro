'''
Created 13 August, 2014
@author : iHub Research

'''

# load required libraries
from __future__ import division
import image_processing
import numpy as np
from flask import Flask
import json


# create flask web server 
app = Flask(__name__)

# capture HTTP endpoint
@app.route('/ImPro/<road>')

def get_route(road):
	# create Directory
	#road = 'wilson'

	route = image_processing.Route(road)
	route.set_dir()

	# grab Image  
	route.images()

	# load images
	x =route.load()

	# get number of different pixels 
	y = route.diffImg(x['img_a.jpg'],x['img_b.jpg'],x['img_c.jpg'])
	print y

	# get distance moved by tracked pixles
	z = route.opticalFlow(x['img_a.jpg'],x['img_b.jpg'],x['img_c.jpg'])
	print z

	# distance
	d = route.distance(z[0][0],z[1][0])
	print d

	if d > 100: 
		update = "traffic is clear on" + road
	else: 
		update = "traffic is standstill on" + road

	updates=json.dumps(update)
	return updates

if __name__ == "__main__":
    	app.run()


   



