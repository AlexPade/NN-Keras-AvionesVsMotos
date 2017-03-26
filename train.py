import h5py
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential

#Recuperamos los datos desde data.h5
datasets = h5py.File('data.h5', "r")
X_train = datasets["train_set_x"][:,:]
y_train = datasets["train_set_y"]
X_test = datasets["valid_set_x"][:,:]
y_test = datasets["valid_set_y"]


n_train, height, width, _, = X_train.shape
n_test, _, _, _,= X_test.shape


X_train = X_train.reshape(n_train, height, width, 3).astype('float32')
X_test = X_test.reshape(n_test, height, width, 3).astype('float32')

#Normalizamos
X_train /= 255
X_test /= 255

#2 clases, aviones y motos
n_class = 2

y_train = to_categorical(y_train,n_class)
y_test = to_categorical(y_test,n_class)

#modelo secuencial
model = Sequential()


n_filters = 32

n_conv = 3

n_pool = 2


from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D

model.add(Convolution2D(
		n_filters, n_conv, n_conv,
		border_mode='valid',
		input_shape=(height,width,3)
	)) 

model.add(Activation('relu'))
#Capas del modelo:
model.add(Convolution2D(n_filters,n_conv,n_conv))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(n_pool,n_pool)))

from keras.layers import Dropout,Flatten, Dense
model.add(Dropout(0.5))

model.add(Flatten()) 

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(2))
model.add(Activation('sigmoid'))

#Compilamos el modelo
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#numero de ejemplos a mirar para cada iteracion
batch_size = 128
#numero de repeticiones del entrenamiento
n_epochs = 20


model.fit(X_train,y_train,
	batch_size=batch_size,
	nb_epoch=n_epochs,
	validation_data=(X_test,y_test))

#evaluamos el modelo
loss,accuracy = model.evaluate(X_test,y_test)
print('loss:', loss)
print('accuracy:', accuracy)

#convertimos el modelo a json
model_json = model.to_json()
with open("model.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")
print("saved model to disk")
