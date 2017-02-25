#the purpose of this file os to setup a basic framework for parsing the image directory (and subdirectories)
#and build a dataframe with the listing_id (aka the directory names and picture file 'prefix') to be merged with the main dataframe

import os
import pandas as pd

imgdir = "../data/images_sample"
img_list = []

try:
    set(os.listdir(imgdir))
except Exception, e:
    print "error with: " + imgdir

for listing in set(os.listdir(imgdir)):
    if listing == '.DS_Store':
        break
    else:
        print "Opening listing : " + listing + '\n'
