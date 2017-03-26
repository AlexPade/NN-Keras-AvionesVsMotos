#################################################
#											    #
#Crea la data en .h5 de las imagenes del dataset#
#												#
#################################################

#python directorios.py /ruta/de/la/carpeta/aTransformar 

import os
import sys
import h5py
from scipy import ndimage
from scipy.misc import imresize
import numpy as np
import cv2

if len(sys.argv) < 2:
    print "Usage: python gen_h5.py input_folder"
    exit(1)

# carpeta de entrada
fi = sys.argv[1]


classes = os.listdir(fi)
set_x = []
set_y = []
k = 0 
list_classes = []

# Create sets
for cls in classes:
	list_classes.append(cls)
	imgs = os.listdir(fi + cls)
	for img in imgs:
		im = cv2.imread(fi + cls + '/' + img)	
		set_x.append(im)
		set_y.append(k)
	k +=1

print(set_x[8])
train_set_x = [set_x[i] for i in range(len(set_x)-150)]
train_set_y = [set_y[i] for i in range(len(set_y)-150)]
valid_set_x = [set_x[i] for i in range(len(set_x)-150, len(set_x))]
valid_set_y = [set_y[i] for i in range(len(set_y)-150, len(set_y))]

valid_set_x = np.array(valid_set_x)
valid_set_y = np.array(valid_set_y)
train_set_x = np.array(train_set_x)
train_set_y = np.array(train_set_y)



f = h5py.File('data.h5','w')
f.create_dataset('train_set_x', data=train_set_x)
f.create_dataset('train_set_y', data=train_set_y)
f.create_dataset('valid_set_x', data=valid_set_x)
f.create_dataset('valid_set_y', data=valid_set_y)
f.create_dataset('list_classes', data=list_classes)