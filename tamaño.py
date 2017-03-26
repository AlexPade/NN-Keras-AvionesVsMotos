

import os
import sys
import h5py
from PIL import Image
from scipy import ndimage
from scipy.misc import imresize
import numpy as np
import scipy.ndimage

if len(sys.argv) < 2:
    print "Ingrese ruta de la carpeta a transformar"
    exit(1)

fi = sys.argv[1]

newDir = sys.argv[2]

imgs = os.listdir(fi)

os.mkdir(newDir)

for img in imgs:
	im = Image.open(fi + '/' + img)
	im = im.resize((64, 64), Image.ANTIALIAS)
	im.save(newDir + '/' + img)
	

