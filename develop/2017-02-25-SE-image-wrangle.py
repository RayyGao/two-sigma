#the purpose of this file os to setup a basic framework for parsing the image directory (and subdirectories)
#and build a dataframe with the listing_id (aka the directory names and picture file 'prefix') to be merged with the main dataframe
import numpy as np
import os
import pandas as pd
from PIL import Image
imgdir = "../data/images_sample/"
img_list = []

try:
    set(os.listdir(imgdir))
except Exception, e:
    print "error with: " + imgdir

for listing in set(os.listdir(imgdir)):
    img_dict = {}
    listingdir = imgdir + listing
    if listing == '.DS_Store':
        break
    else:
        print "Opening listing : " + listing + '\n'
        img_dict['listing_id'] = listing
        img_dict['img_quantity'] = len(os.listdir(listingdir))
        print len(os.listdir(listingdir))
        for img in set(os.listdir(listingdir)):
                       if img.find('.jpg') > 0:
                           print img
                           imag = Image.open(imgdir + listing + '/'+ img)
                           imag = imag.convert('RGB')
                           X,Y = 0,0
                           pixelRGB = imag.getpixel((X,Y))
                           R,G,B = pixelRGB
                           brightness = sum([R,G,B])/3
                           print "Brightness is: " + str(brightness) + "\n"
                           #Standard
                           LuminanceA = (0.2126*R) + (0.7152*G) + (0.0722*B)
                           print "Luminance A is: " + str(LuminanceA) + "\n"
                           #Percieved A
                           LuminanceB = (0.299*R + 0.587*G + 0.114*B)
                           #Perceived B, slower to calculate
                           print "Luminance B is: " + str(LuminanceB) + "\n"
                           LuminanceC =np.sqrt( 0.241*R*R + 0.691*G*G + 0.068*B*B )
                           print "Luminance C is: " + str(LuminanceC) + "\n"
