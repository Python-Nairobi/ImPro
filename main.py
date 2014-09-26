'''
Created 13 August, 2014
@author : iHub Research

'''
# load required libraries
import image_processing
import numpy as np
from flask import Flask

# create flask web server
app = Flask(__name__)

# capture HTTP endpoint
@app.route('/ImPro/<road>')

def get_route(road):
	# create Directory
	route = image_processing.Route(road)
	route.set_dir()

	# grab Image  
	route.images()

	# load images
	x =route.load()

	# motion detection
	y = route.motion(x['img_a.jpg'],x['img_b.jpg'],x['img_c.jpg'])

	return y

if __name__ == "__main__":
    app.run(host='192.168.33.71')


