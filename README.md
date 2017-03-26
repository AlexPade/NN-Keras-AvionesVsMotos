# NN-Keras-AvionesVsMotos
Red Neuronal utilizando Keras, para la clasificacion de aviones y motos

en data/ -> imagenes que son utilizadas para el entrenamiento.

en test/ -> imagenes que son utilizadas para testear la NN.

directorios.py -> crea el archivo data.h5 con los datos de las imagenes guardadas en data/

	#python directorios.py /ruta/data/ 

train.py -> entrena la red neuronal, usando los datos almacenados en data.h5

	#python train.py data.h5

test.py -> testea la red neuronal con las imagenes presentes en test/

	#python test.py
