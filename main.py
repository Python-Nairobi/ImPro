'''
Created 13 August, 2014
@author : iHub Research

'''
import image_processing
import numpy as np

# create Directory
route = image_processing.Route('nhif')
route.set_dir()

# grab Image  
route.images()

# load images
x =route.load()
print len(x)
print type(x)

# motion detection
y = route.motion(x['img_a.jpg'],x['img_b.jpg'],x['img_c.jpg'])
print y

