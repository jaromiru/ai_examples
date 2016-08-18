# MNIST recognition implementation with Keras
# Convolutional NN
# ----------------------------------------------------

import os    
os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"   

from keras.models import Sequential, load_model
from keras.layers import *
from keras.utils import np_utils
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32') / 255
y_train = np_utils.to_categorical(y_train, 10)

X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32') / 255
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Convolution2D(16, 5, 5, border_mode="same", input_shape=(1, 28, 28)))
model.add(Activation("relu"))

model.add(MaxPooling2D())

model.add(Convolution2D(8, 3, 3, border_mode="same"))
model.add(Activation("relu"))

model.add(Convolution2D(8, 3, 3, border_mode="same"))
model.add(Activation("relu"))

model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(output_dim=128))
model.add(Activation("relu"))

model.add(Dense(output_dim=64))
model.add(Activation("relu"))

model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for i in range(5):
	model.fit(X_train, y_train, nb_epoch=5, batch_size=100) 

	loss_and_metrics = model.evaluate(X_test, y_test, batch_size=500)
	print(loss_and_metrics)

	model.save('model_mnist.h5')