import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import os
import sys
import cv2

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#cargamos el modelo
loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

#compilamos
loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


def tipo(numero):
	if(numero==1):
		toreturn="avion"
	if(numero==0):
		toreturn="moto"
	return toreturn
i=1


#Testeamos con la carpeta test/ 

fi="test/"

imgs = os.listdir(fi)

for img in imgs:
    x = imread(fi + img)
    x = cv2.resize(x,(64,64))
    x = x.reshape(1,64,64,3)

    out = loaded_model.predict(x)
    #print(out)
    print(img)
    print(tipo(np.argmax(out,axis=1)))

